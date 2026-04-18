"""
Fine-tune PromoterAI on GTEx rare variant outliers using TwinModel.

Usage:
    python -m torch_promoterai.finetune \
        --model_checkpoint <path/to/best_model.pt> \
        --var_file data/annotation/finetune_gtex.tsv \
        --fasta_file <genome.fa> \
        --input_length 20480 --batch_size 8
"""

import argparse
import os

import pandas as pd
import pyfaidx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_promoterai.architecture import TwinModel
from torch_promoterai.dataset import VariantDataset
from torch_promoterai.utils import (
    CSVLogger,
    WeightDecayScheduler,
    load_pretrained,
    make_lr_lambda,
)


def _collate_variant(batch):
    """Stack ref/alt tensors and labels from a list of VariantDataset items into batched tensors."""
    x_refs = torch.stack([item[0][0] for item in batch])
    x_alts = torch.stack([item[0][1] for item in batch])
    ys = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    return (x_refs, x_alts), ys


def _run_epoch(twin_model, loader, optimizer, device, max_steps=None, train=True):
    """Run one train or eval epoch; stops after max_steps batches when set."""
    twin_model.train(train)
    total_loss, n_steps = 0.0, 0
    with torch.set_grad_enabled(train):
        for (x_ref, x_alt), y in loader:
            x_ref, x_alt, y = x_ref.to(device), x_alt.to(device), y.to(device)
            diff = twin_model(x_ref, x_alt)
            loss = F.mse_loss(diff, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(twin_model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_steps += 1
            if max_steps and n_steps >= max_steps:
                break
    return total_loss / max(n_steps, 1)


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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    twin_model_folder = args.model_checkpoint.replace(".pt", "") + "_finetune"
    os.makedirs(twin_model_folder, exist_ok=True)

    base_model, _ = load_pretrained(args.model_checkpoint, map_location=str(device))
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
        df_train, fasta, args.input_length, output_col="z", shuffle=True
    )
    ds_valid = VariantDataset(df_valid, fasta, args.input_length, output_col="z")

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
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(args.epochs)
    )
    wd_scheduler = WeightDecayScheduler(optimizer, args.weight_decay, args.epochs)

    # steps_per_epoch = 20% of training data (matching TF len(gen_train) // 5)
    steps_per_epoch = max(1, len(ds_train) // (5 * args.batch_size))
    logger = CSVLogger(os.path.join(twin_model_folder, "logs.csv"))
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = _run_epoch(
            twin_model,
            loader_train,
            optimizer,
            device,
            max_steps=steps_per_epoch,
            train=True,
        )
        val_loss = _run_epoch(twin_model, loader_valid, optimizer, device, train=False)

        scheduler.step()
        wd_scheduler.step(epoch)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={lr_now:.2e}"
        )
        logger.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr_now,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save base model (not twin wrapper), matching TF CustomModelCheckpoint behavior
            torch.save(
                {"model_state_dict": twin_model.base_model.state_dict()},
                os.path.join(twin_model_folder, "best_model.pt"),
            )
            print(f"  Saved best model (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()
