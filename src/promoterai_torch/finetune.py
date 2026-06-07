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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler

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
    resolve_per_rank_batch_size,
    setup_distributed,
    unwrap_model,
)


def _collate_variant(batch):
    """Stack ref/alt tensors and labels from a list of VariantDataset items into batched tensors."""
    x_refs = torch.stack([item[0][0] for item in batch])
    x_alts = torch.stack([item[0][1] for item in batch])
    ys = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    return (x_refs, x_alts), ys


class DistributedSliceSampler(Sampler):
    """Shard a fixed prefix across ranks without padding or duplication."""

    def __init__(self, total_size: int, rank: int, world_size: int):
        self.total_size = total_size
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, self.total_size, self.world_size))

    def __len__(self):
        return self.total_size // self.world_size


def resolve_finetune_epoch_sizes(
    train_size: int, val_size: int, global_batch_size: int
) -> tuple[int, int]:
    """Return official steps/epoch and validation prefix size for complete batches."""
    steps_per_epoch = (train_size // global_batch_size) // 5
    if steps_per_epoch < 1:
        raise ValueError(
            "Finetuning requires at least five complete global batches; "
            f"got {train_size} variants with --batch_size {global_batch_size}"
        )
    complete_val_samples = (val_size // global_batch_size) * global_batch_size
    if complete_val_samples == 0:
        raise ValueError(
            "Validation requires at least one complete global batch; "
            f"got {val_size} variants with --batch_size {global_batch_size}"
        )
    return steps_per_epoch, complete_val_samples


def _clip_grad_norm_per_parameter(parameters, max_norm: float) -> None:
    """Match Keras AdamW(clipnorm=...) by clipping each variable independently."""
    for parameter in parameters:
        if parameter.grad is not None:
            nn.utils.clip_grad_norm_([parameter], max_norm=max_norm)


def build_finetune_optimizer(twin_model, learning_rate: float, weight_decay: float):
    """Build the Keras-compatible AdamW optimizer for the trainable output head."""
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, twin_model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-7,
    )


def _run_epoch(
    twin_model,
    loader,
    optimizer,
    device,
    max_steps=None,
    train=True,
    amp_dtype=None,
    grad_scaler=None,
    world_size=1,
):
    """Run one train or eval epoch; stops after max_steps batches when set."""
    twin_model.train(train)
    total_loss, n_samples, n_steps = 0.0, 0, 0
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
                    _clip_grad_norm_per_parameter(
                        twin_model.parameters(), max_norm=1.0
                    )
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    _clip_grad_norm_per_parameter(
                        twin_model.parameters(), max_norm=1.0
                    )
                    optimizer.step()

            batch_size = int(y.shape[0])
            total_loss += loss.item() * batch_size
            n_samples += batch_size
            n_steps += 1
            if max_steps and n_steps >= max_steps:
                break
    loss_stats = torch.tensor(
        [total_loss, n_samples], dtype=torch.float64, device=device
    )
    if world_size > 1:
        dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
    reduced_loss, reduced_samples = loss_stats.tolist()
    return reduced_loss / max(reduced_samples, 1.0)


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
    inference_only: bool = False,
):
    """Save an inference-only model or a full resumable finetuning checkpoint."""
    twin_model = unwrap_model(twin_model)
    args_dict = dict(base_args)
    args_dict.setdefault(
        "output_crop", args_dict.get("input_length", 0) - args_dict.get("output_length", 0)
    )
    args_dict["finetune_args"] = dict(finetune_args)
    checkpoint = {
        "model_state_dict": twin_model.base_model.state_dict(),
        "args": args_dict,
    }
    if not inference_only:
        checkpoint.update(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": None,
                "grad_scaler_state_dict": (
                    grad_scaler.state_dict()
                    if grad_scaler is not None and grad_scaler.is_enabled()
                    else None
                ),
                "val_loss": val_loss,
                "best_val_loss": (
                    val_loss if best_val_loss is None else best_val_loss
                ),
                "epoch": epoch,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
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
    twin_model = unwrap_model(twin_model)
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Global batch size; divided evenly across torchrun ranks",
    )
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

    rank, world_size, device = setup_distributed()
    per_rank_batch_size = resolve_per_rank_batch_size(args.batch_size, world_size)
    twin_model_folder = args.model_checkpoint.replace(".pt", "") + "_finetune"
    if rank == 0:
        os.makedirs(twin_model_folder, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    base_model, base_args = load_pretrained(args.model_checkpoint, map_location=str(device))
    twin_model = TwinModel(base_model).to(device)
    if world_size > 1:
        twin_model = DistributedDataParallel(
            twin_model,
            device_ids=[device.index] if device.type == "cuda" else None,
        )

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

    steps_per_epoch, complete_val_samples = resolve_finetune_epoch_sizes(
        len(ds_train), len(ds_valid), args.batch_size
    )

    train_sampler = (
        DistributedSampler(
            ds_train,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        if world_size > 1
        else None
    )
    val_sampler = (
        DistributedSliceSampler(complete_val_samples, rank, world_size)
        if world_size > 1
        else None
    )
    loader_train = DataLoader(
        ds_train,
        batch_size=per_rank_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=_collate_variant,
        drop_last=True,
    )
    loader_valid = DataLoader(
        ds_valid,
        batch_size=per_rank_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=_collate_variant,
        drop_last=True,
    )

    # Only the unfrozen output_heads[0] params have gradients
    optimizer = build_finetune_optimizer(
        twin_model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    amp_torch_dtype = resolve_amp_dtype(args.amp_dtype)
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp_dtype == "fp16" and device.type == "cuda"
    )
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
        if rank == 0:
            print(
                f"Resumed finetuning from {resume_checkpoint} at epoch "
                f"{start_epoch + 1}; best_val_loss={best_val_loss:.4f}"
            )
    wandb_config = dict(base_args)
    wandb_config["finetune_args"] = vars(args)
    wandb_config["n_train_variants"] = len(ds_train)
    wandb_config["n_val_variants"] = len(ds_valid)
    wandb_config["start_epoch"] = start_epoch
    wandb_config["global_batch_size"] = args.batch_size
    wandb_config["per_rank_batch_size"] = per_rank_batch_size
    wandb_config["world_size"] = world_size
    wandb_run = init_wandb(args, wandb_config, rank=rank)

    if rank == 0:
        print(
            "Finetuning setup: "
            f"world_size={world_size} global_batch_size={args.batch_size} "
            f"per_rank_batch_size={per_rank_batch_size} "
            f"steps_per_epoch={steps_per_epoch} amp_dtype={args.amp_dtype}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
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
            world_size=world_size,
        )
        val_loss = _run_epoch(
            twin_model,
            loader_valid,
            optimizer,
            device,
            train=False,
            amp_dtype=amp_torch_dtype,
            world_size=world_size,
        )

        lr_now = optimizer.param_groups[0]["lr"]
        wd_now = optimizer.param_groups[0]["weight_decay"]
        checkpoint_saved = val_loss < best_val_loss
        next_best_val_loss = min(best_val_loss, val_loss)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr_now,
            "wd": wd_now,
            "best_val_loss": next_best_val_loss,
            "checkpoint_saved": int(checkpoint_saved),
        }
        if checkpoint_saved:
            best_val_loss = next_best_val_loss

        if rank == 0:
            print(
                f"Epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  lr={lr_now:.2e}  wd={wd_now:.2e}"
            )
            logger.log(row)
            log_wandb(wandb_run, row, step=epoch + 1)

            if checkpoint_saved:
                save_finetune_checkpoint(
                    twin_model,
                    optimizer,
                    val_loss,
                    epoch,
                    twin_model_folder,
                    base_args,
                    {
                        **vars(args),
                        "global_batch_size": args.batch_size,
                        "per_rank_batch_size": per_rank_batch_size,
                        "world_size": world_size,
                    },
                    best_val_loss=best_val_loss,
                    grad_scaler=grad_scaler,
                    inference_only=True,
                )
                print(f"  Saved best model (val_loss={val_loss:.4f})")

            save_finetune_checkpoint(
                twin_model,
                optimizer,
                val_loss,
                epoch,
                twin_model_folder,
                base_args,
                {
                    **vars(args),
                    "global_batch_size": args.batch_size,
                    "per_rank_batch_size": per_rank_batch_size,
                    "world_size": world_size,
                },
                checkpoint_name="latest_model.pt",
                best_val_loss=best_val_loss,
                grad_scaler=grad_scaler,
                atomic=True,
            )
        if world_size > 1:
            dist.barrier()

    finish_wandb(wandb_run)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
