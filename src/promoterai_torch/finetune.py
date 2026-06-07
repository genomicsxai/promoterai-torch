"""
Fine-tune PromoterAI on GTEx rare variant outliers using TwinModel.

Usage:
    python -m promoterai_torch.finetune \
        --model_checkpoint <path/to/best_model.pt> \
        --var_file data/annotation/finetune_gtex.tsv \
        --fasta_file <genome.fa> \
        --input_length 20480 --batch_size 8
"""

import argparse
import os
import tempfile

import pandas as pd
import pyfaidx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from promoterai_torch.architecture import TwinModel
from promoterai_torch.dataset import VariantDataset
from promoterai_torch.utils import (
    CSVLogger,
    add_wandb_args,
    apply_optimizer_schedule,
    autocast_context,
    finish_wandb,
    init_wandb,
    log_wandb,
    load_pretrained,
    normalize_model_state_dict,
    resolve_amp_dtype,
)


def _collate_variant(batch):
    """Stack ref/alt tensors and labels from a list of VariantDataset items into batched tensors."""
    x_refs = torch.stack([item[0][0] for item in batch])
    x_alts = torch.stack([item[0][1] for item in batch])
    ys = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    return (x_refs, x_alts), ys


def _run_epoch(
    twin_model,
    loader,
    optimizer,
    device,
    max_steps=None,
    train=True,
    amp_dtype=None,
    grad_scaler=None,
):
    """Run one train or eval epoch; stops after max_steps batches when set."""
    twin_model.train(train)
    total_loss, n_steps = 0.0, 0
    with torch.set_grad_enabled(train):
        for (x_ref, x_alt), y in loader:
            x_ref, x_alt, y = x_ref.to(device), x_alt.to(device), y.to(device)
            with autocast_context(device, amp_dtype):
                diff = twin_model(x_ref, x_alt)
                loss = F.mse_loss(diff, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if grad_scaler is not None and grad_scaler.is_enabled():
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(twin_model.parameters(), max_norm=1.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(twin_model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item()
            n_steps += 1
            if max_steps and n_steps >= max_steps:
                break
    return total_loss / max(n_steps, 1)


def save_finetune_checkpoint(
    twin_model,
    optimizer,
    val_loss: float,
    epoch: int,
    checkpoint_folder: str,
    base_args: dict,
    finetune_args: dict,
    checkpoint_name: str = "best_model.pt",
    best_val_loss: float | None = None,
    grad_scaler=None,
    atomic: bool = False,
):
    """Save a load_pretrained-compatible base-model checkpoint from a TwinModel."""
    args_dict = dict(base_args)
    args_dict.setdefault(
        "output_crop", args_dict.get("input_length", 0) - args_dict.get("output_length", 0)
    )
    args_dict["finetune_args"] = dict(finetune_args)
    checkpoint = {
        "model_state_dict": twin_model.base_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "grad_scaler_state_dict": (
            grad_scaler.state_dict()
            if grad_scaler is not None and grad_scaler.is_enabled()
            else None
        ),
        "val_loss": val_loss,
        "best_val_loss": val_loss if best_val_loss is None else best_val_loss,
        "epoch": epoch,
        "args": args_dict,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_name)
    if not atomic:
        torch.save(checkpoint, checkpoint_path)
        return

    fd, temp_path = tempfile.mkstemp(
        dir=checkpoint_folder, prefix=f".{checkpoint_name}.", suffix=".tmp"
    )
    os.close(fd)
    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def resolve_finetune_resume_checkpoint(
    resume_checkpoint: str | None,
    auto_resume: bool,
    checkpoint_folder: str,
) -> str | None:
    """Resolve explicit or automatic finetuning checkpoint resumption."""
    if resume_checkpoint:
        return resume_checkpoint
    if not auto_resume:
        return None
    latest_checkpoint = os.path.join(checkpoint_folder, "latest_model.pt")
    if os.path.exists(latest_checkpoint):
        return latest_checkpoint
    return None


def load_finetune_checkpoint(
    twin_model,
    optimizer,
    grad_scaler,
    checkpoint_path: str,
    device,
) -> tuple[int, float, dict]:
    """Restore finetuning state and return (start_epoch, best_val_loss, args)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    twin_model.base_model.load_state_dict(
        normalize_model_state_dict(checkpoint["model_state_dict"])
    )
    if checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler_state = checkpoint.get("grad_scaler_state_dict")
    if (
        grad_scaler is not None
        and grad_scaler.is_enabled()
        and scaler_state is not None
    ):
        grad_scaler.load_state_dict(scaler_state)
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    cuda_rng_state = checkpoint.get("cuda_rng_state_all")
    if cuda_rng_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_state)

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_val_loss = float(
        checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))
    )
    return start_epoch, best_val_loss, checkpoint.get("args", {})


def main():
    """Filter GTEx outlier variants, fine-tune TwinModel, save best base model checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--var_file", required=True)
    parser.add_argument("--fasta_file", required=True)
    parser.add_argument("--input_length", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--amp_dtype", choices=("none", "bf16", "fp16"), default="none"
    )
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        default=False,
        help="Resume from the finetune output folder's latest_model.pt when present",
    )
    add_wandb_args(parser)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    twin_model_folder = args.model_checkpoint.replace(".pt", "") + "_finetune"
    os.makedirs(twin_model_folder, exist_ok=True)

    base_model, base_args = load_pretrained(args.model_checkpoint, map_location=str(device))
    twin_model = TwinModel(base_model).to(device)

    df_var = pd.read_csv(args.var_file, sep="\t")
    df_var = df_var[(df_var["in_cds"] == 0) & (df_var["spliceai"] < 0.05)]
    df_outlier = df_var[(df_var["p_under"] < 0.01) | (df_var["p_over"] < 0.01)]
    df_var = df_var[df_var["gene"].isin(df_outlier["gene"])]

    train_chroms = [f"chr{i}" for i in range(1, 21, 2)]
    val_chroms = [f"chr{i}" for i in range(21, 23)]
    df_train = df_var[df_var["chrom"].isin(train_chroms)]
    df_valid = df_var[df_var["chrom"].isin(val_chroms)]

    fasta = pyfaidx.Fasta(args.fasta_file)
    ds_train = VariantDataset(
        df_train, fasta, args.input_length, output_col="z", boundary="zeros"
    )
    ds_valid = VariantDataset(
        df_valid, fasta, args.input_length, output_col="z", boundary="zeros"
    )

    loader_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate_variant,
    )
    loader_valid = DataLoader(
        ds_valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_variant,
    )

    # Only the unfrozen output_heads[0] params have gradients
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, twin_model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    amp_torch_dtype = resolve_amp_dtype(args.amp_dtype)
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp_dtype == "fp16" and device.type == "cuda"
    )
    # steps_per_epoch = 20% of training data (matching TF len(gen_train) // 5)
    steps_per_epoch = max(1, len(ds_train) // (5 * args.batch_size))
    logger = CSVLogger(os.path.join(twin_model_folder, "logs.csv"))
    start_epoch = 0
    best_val_loss = float("inf")
    resume_checkpoint = resolve_finetune_resume_checkpoint(
        args.resume_checkpoint, args.auto_resume, twin_model_folder
    )
    if resume_checkpoint:
        start_epoch, best_val_loss, _ = load_finetune_checkpoint(
            twin_model,
            optimizer,
            grad_scaler,
            resume_checkpoint,
            device,
        )
        print(
            f"Resumed finetuning from {resume_checkpoint} at epoch "
            f"{start_epoch + 1}; best_val_loss={best_val_loss:.4f}"
        )
    wandb_config = dict(base_args)
    wandb_config["finetune_args"] = vars(args)
    wandb_config["n_train_variants"] = len(ds_train)
    wandb_config["n_val_variants"] = len(ds_valid)
    wandb_config["start_epoch"] = start_epoch
    wandb_run = init_wandb(args, wandb_config)

    for epoch in range(start_epoch, args.epochs):
        apply_optimizer_schedule(
            optimizer, args.learning_rate, args.weight_decay, args.epochs, epoch
        )
        train_loss = _run_epoch(
            twin_model,
            loader_train,
            optimizer,
            device,
            max_steps=steps_per_epoch,
            train=True,
            amp_dtype=amp_torch_dtype,
            grad_scaler=grad_scaler,
        )
        val_loss = _run_epoch(
            twin_model,
            loader_valid,
            optimizer,
            device,
            train=False,
            amp_dtype=amp_torch_dtype,
        )

        lr_now = optimizer.param_groups[0]["lr"]
        wd_now = optimizer.param_groups[0]["weight_decay"]
        checkpoint_saved = val_loss < best_val_loss
        print(
            f"Epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={lr_now:.2e}  wd={wd_now:.2e}"
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr_now,
            "wd": wd_now,
            "best_val_loss": min(best_val_loss, val_loss),
            "checkpoint_saved": int(checkpoint_saved),
        }
        logger.log(row)
        log_wandb(wandb_run, row, step=epoch + 1)

        if checkpoint_saved:
            best_val_loss = val_loss
            save_finetune_checkpoint(
                twin_model,
                optimizer,
                val_loss,
                epoch,
                twin_model_folder,
                base_args,
                vars(args),
                best_val_loss=best_val_loss,
                grad_scaler=grad_scaler,
            )
            print(f"  Saved best model (val_loss={val_loss:.4f})")

        save_finetune_checkpoint(
            twin_model,
            optimizer,
            val_loss,
            epoch,
            twin_model_folder,
            base_args,
            vars(args),
            checkpoint_name="latest_model.pt",
            best_val_loss=best_val_loss,
            grad_scaler=grad_scaler,
            atomic=True,
        )

    finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
