"""
Multi-GPU training script (torchrun-compatible).

Single-GPU:
    python -m promoterai_torch.train [args]

Multi-GPU:
    torchrun --nproc_per_node=4 -m promoterai_torch.train [args]
"""

import argparse
import os
from glob import glob

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from promoterai_torch.architecture import PromoterAI
from promoterai_torch.dataset import SequenceDataset, build_weighted_dataloader
from promoterai_torch.utils import (
    CSVLogger,
    apply_optimizer_schedule,
    save_checkpoint,
)


def setup_distributed():
    """Initialize DDP if LOCAL_RANK is set; returns (rank, world_size, device)."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), torch.device(f"cuda:{local_rank}")


def build_model(args, output_dims, device, world_size, rank):
    """Build PromoterAI, convert BN to SyncBatchNorm, and wrap in DDP when world_size > 1."""
    model = PromoterAI(
        num_blocks=args.num_blocks,
        model_dim=args.model_dim,
        output_dims=output_dims,
        output_crop=args.input_length - args.output_length,
    ).to(device)
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[rank])
    return model


def compute_loss(outputs, y_tuple, w_tuple):
    """Compute per-species weighted MSE loss; skips species whose y has only 1 track (dummy)."""
    loss = torch.tensor(0.0, device=outputs[0].device)
    for j, (y_pred, y_true, w) in enumerate(zip(outputs, y_tuple, w_tuple)):
        if y_true.shape[-1] == 1:
            continue  # dummy target for non-matching species
        per_sample = F.mse_loss(y_pred, y_true, reduction="none").mean(
            dim=(1, 2)
        )  # (B,)
        loss = loss + (per_sample * w[:, j]).mean()
    return loss


def _run_epoch(model, loader, optimizer, device, world_size=1, max_steps=None, train=True):
    """Run one train or eval epoch; stops after max_steps batches when set."""
    model.train(train)
    total_loss, n_steps = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            x, y_tuple, w_tuple = batch
            x = x.to(device)
            y_tuple = tuple(y.to(device) for y in y_tuple)
            w_tuple = w_tuple.to(device)

            outputs = model(x)
            loss = compute_loss(outputs, y_tuple, w_tuple)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-4)
                optimizer.step()

            total_loss += loss.item()
            n_steps += 1
            if max_steps and n_steps >= max_steps:
                break

    stats = torch.tensor([total_loss, n_steps], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item() / max(stats[1].item(), 1.0))


def main():
    """Parse args, build datasets and model, run training loop, save best checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_folder", required=True)
    parser.add_argument("--hdf5_human_folder", required=True)
    parser.add_argument("--hdf5_nonhuman_folders", nargs="+", default=[])
    parser.add_argument("--input_length", type=int, required=True)
    parser.add_argument("--output_length", type=int, required=True)
    parser.add_argument("--num_blocks", type=int, required=True)
    parser.add_argument("--model_dim", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    rank, world_size, device = setup_distributed()

    hdf5_folders = [args.hdf5_human_folder] + args.hdf5_nonhuman_folders
    num_species = len(hdf5_folders)

    # Build datasets — human: chr1-20 train, chr21-22 val; non-human: all chroms
    train_datasets, val_dataset_files = [], None
    for j, folder in enumerate(hdf5_folders):
        sw = tuple(k == j for k in range(num_species))
        if j == 0:
            train_files = sum([glob(f"{folder}/chr{i}_*") for i in range(1, 21)], [])
            val_files = sum([glob(f"{folder}/chr{i}_*") for i in range(21, 23)], [])
            val_dataset_files = val_files
        else:
            train_files = glob(f"{folder}/chr*")
        train_datasets.append(
            SequenceDataset(
                train_files, args.input_length, args.output_length, sw, augment=True
            )
        )

    # Output dims inferred from first batch of each dataset
    output_dims = []
    for j, ds in enumerate(train_datasets):
        _, y_tuple, _ = ds[0]
        output_dims.append(y_tuple[j].shape[-1])

    dataset_sizes = [len(d) for d in train_datasets]
    steps_per_epoch = int(sum(dataset_sizes) / 10)
    train_samples_per_rank = steps_per_epoch * args.batch_size

    val_sw = tuple(k == 0 for k in range(num_species))
    val_dataset = SequenceDataset(
        val_dataset_files, args.input_length, args.output_length, val_sw
    )

    train_loader = build_weighted_dataloader(
        train_datasets,
        args.batch_size,
        args.num_workers,
        rank,
        world_size,
        num_samples=train_samples_per_rank,
    )
    val_loader = build_weighted_dataloader(
        [val_dataset],
        args.batch_size,
        args.num_workers,
        rank,
        world_size,
        shuffle=False,
    )

    model = build_model(args, output_dims, device, world_size, rank)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    os.makedirs(args.checkpoint_folder, exist_ok=True)
    logger = CSVLogger(os.path.join(args.checkpoint_folder, "logs.csv"))
    best_val_loss = float("inf")

    args_dict = vars(args)
    args_dict["output_dims"] = output_dims
    args_dict["output_crop"] = args.input_length - args.output_length

    for epoch in range(args.epochs):
        apply_optimizer_schedule(
            optimizer, args.learning_rate, args.weight_decay, args.epochs, epoch
        )
        train_loss = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            world_size=world_size,
            max_steps=steps_per_epoch,
            train=True,
        )
        val_loss = _run_epoch(
            model, val_loader, optimizer, device, world_size=world_size, train=False
        )

        if rank == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            wd_now = optimizer.param_groups[0]["weight_decay"]
            print(
                f"Epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={lr_now:.2e}  wd={wd_now:.2e}"
            )
            logger.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr_now,
                    "wd": wd_now,
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    None,
                    val_loss,
                    epoch,
                    args.checkpoint_folder,
                    args_dict,
                )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
