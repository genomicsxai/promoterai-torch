"""
Multi-GPU training script (torchrun-compatible).

Single-GPU:
    python -m promoterai_torch.train [args]

Multi-GPU:
    torchrun --nproc_per_node=4 -m promoterai_torch.train [args]
"""

import argparse
import os
import time
from contextlib import nullcontext
from glob import glob

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from promoterai_torch.architecture import PromoterAI
from promoterai_torch.dataset import SequenceDataset, build_weighted_dataloader
from promoterai_torch.utils import (
    CSVLogger,
    add_wandb_args,
    apply_optimizer_schedule,
    finish_wandb,
    init_wandb,
    log_wandb,
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
    """Build PromoterAI and optionally wrap it for distributed training."""
    model = PromoterAI(
        num_blocks=args.num_blocks,
        model_dim=args.model_dim,
        output_dims=output_dims,
        output_crop=args.input_length - args.output_length,
    ).to(device)
    if world_size > 1 and not args.no_sync_batchnorm:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.compile:
        model = torch.compile(model)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank])
    return model


def resolve_per_rank_batch_size(global_batch_size: int, world_size: int) -> int:
    """Return the rank-local batch size for a requested global batch size."""
    if global_batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if global_batch_size % world_size != 0:
        raise ValueError(
            "--batch_size is the global batch size and must be divisible by "
            f"world_size ({world_size}); got {global_batch_size}"
        )
    return global_batch_size // world_size


def resolve_resume_checkpoint(args) -> str | None:
    """Resolve explicit or automatic training checkpoint resumption."""
    if args.resume_checkpoint:
        return args.resume_checkpoint
    if not args.auto_resume:
        return None
    latest_checkpoint = os.path.join(args.checkpoint_folder, "latest_model.pt")
    if os.path.exists(latest_checkpoint):
        return latest_checkpoint
    return None


def load_training_checkpoint(model, optimizer, checkpoint_path: str, device):
    """Restore model/optimizer state and return (start_epoch, best_val_loss, args)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    base = model.module if hasattr(model, "module") else model
    base.load_state_dict(ckpt["model_state_dict"])
    if ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_val_loss = float(ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf"))))
    return start_epoch, best_val_loss, ckpt.get("args", {})


def compute_loss(outputs, y_tuple, w_tuple):
    """Compute per-species weighted MSE loss; skips species whose y has only 1 track (dummy)."""
    loss = torch.tensor(0.0, device=outputs[0].device)
    if w_tuple.dim() == 1:
        w_tuple = w_tuple.unsqueeze(0)
    for j, (y_pred, y_true) in enumerate(zip(outputs, y_tuple)):
        if y_true.shape[-1] == 1:
            continue  # dummy target for non-matching species
        per_sample = F.mse_loss(y_pred, y_true, reduction="none").mean(
            dim=(1, 2)
        )  # (B,)
        loss = loss + (per_sample * w_tuple[:, j]).mean()
    return loss


def resolve_amp_dtype(amp_dtype: str):
    """Map an AMP CLI string to a torch dtype, or None when AMP is disabled."""
    if amp_dtype == "none":
        return None
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported --amp_dtype: {amp_dtype}")


def _sync_for_timing(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast_context(device: torch.device, amp_dtype):
    if amp_dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _reduce_epoch_metrics(metrics: dict, device: torch.device, world_size: int) -> dict:
    """Reduce timing/sample metrics across ranks for rank-local epoch stats."""
    loss_stats = torch.tensor(
        [metrics["total_loss"], metrics["n_steps"]], dtype=torch.float64, device=device
    )
    sum_stats = torch.tensor(
        [
            metrics["local_samples"],
            metrics["profile_samples"],
            metrics["profile_steps"],
            metrics["profile_data_time"],
            metrics["profile_transfer_time"],
            metrics["profile_forward_time"],
            metrics["profile_backward_time"],
        ],
        dtype=torch.float64,
        device=device,
    )
    max_stats = torch.tensor(
        [metrics["epoch_time"], metrics["profile_step_time"]],
        dtype=torch.float64,
        device=device,
    )
    if world_size > 1:
        dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(max_stats, op=dist.ReduceOp.MAX)

    total_loss, total_steps = loss_stats.tolist()
    (
        global_samples,
        profile_samples,
        profile_steps,
        profile_data_time,
        profile_transfer_time,
        profile_forward_time,
        profile_backward_time,
    ) = sum_stats.tolist()
    epoch_time, profile_step_time = max_stats.tolist()

    reduced = {
        "loss": float(total_loss / max(total_steps, 1.0)),
        "global_samples": float(global_samples),
        "epoch_time_sec": float(epoch_time),
        "samples_per_sec": float(global_samples / epoch_time) if epoch_time > 0 else 0.0,
        "profile_samples": float(profile_samples),
        "profile_steps": float(profile_steps),
        "profile_step_time_sec": float(profile_step_time),
        "profile_samples_per_sec": (
            float(profile_samples / profile_step_time) if profile_step_time > 0 else 0.0
        ),
    }
    profile_denominator = max(profile_steps, 1.0)
    reduced.update(
        {
            "profile_data_wait_sec": float(profile_data_time / profile_denominator),
            "profile_transfer_sec": float(profile_transfer_time / profile_denominator),
            "profile_forward_loss_sec": float(
                profile_forward_time / profile_denominator
            ),
            "profile_backward_step_sec": float(
                profile_backward_time / profile_denominator
            ),
        }
    )
    return reduced


def _run_epoch(
    model,
    loader,
    optimizer,
    device,
    world_size=1,
    max_steps=None,
    train=True,
    desc=None,
    show_progress=False,
    log_every_batches=0,
    wandb_run=None,
    wandb_prefix=None,
    wandb_step_offset=0,
    wandb_log_every_batches=0,
    amp_dtype=None,
    grad_scaler=None,
    profile_batches=0,
    profile_warmup_batches=10,
    return_metrics=False,
):
    """Run one train or eval epoch; optionally profile a bounded batch window."""
    model.train(train)
    total_loss, n_steps = 0.0, 0
    local_samples = 0
    profile_steps = 0
    profile_samples = 0
    profile_data_time = 0.0
    profile_transfer_time = 0.0
    profile_forward_time = 0.0
    profile_backward_time = 0.0
    profile_step_time = 0.0
    profile_enabled = profile_batches > 0
    profile_limit = profile_warmup_batches + profile_batches if profile_enabled else None

    total = max_steps
    if profile_limit is not None:
        total = min(total, profile_limit) if total is not None else profile_limit
    if total is None:
        try:
            total = len(loader)
        except TypeError:
            total = None

    batches = tqdm(
        loader,
        desc=desc,
        total=total,
        unit="batch",
        disable=not show_progress,
        leave=False,
    )
    iterator = iter(batches)
    epoch_start = time.perf_counter()
    data_start = time.perf_counter()

    with torch.set_grad_enabled(train):
        while True:
            if max_steps and n_steps >= max_steps:
                break
            if profile_limit is not None and n_steps >= profile_limit:
                break
            try:
                batch = next(iterator)
            except StopIteration:
                break

            batch_ready = time.perf_counter()
            data_wait = batch_ready - data_start
            in_profile_window = (
                profile_enabled
                and n_steps >= profile_warmup_batches
                and profile_steps < profile_batches
            )
            _sync_for_timing(device, in_profile_window)
            step_start = time.perf_counter()

            x, y_tuple, w_tuple = batch
            transfer_start = time.perf_counter()
            x = x.to(device, non_blocking=True)
            y_tuple = tuple(y.to(device, non_blocking=True) for y in y_tuple)
            w_tuple = w_tuple.to(device, non_blocking=True)
            _sync_for_timing(device, in_profile_window)
            transfer_end = time.perf_counter()

            with _autocast_context(device, amp_dtype):
                outputs = model(x)
                loss = compute_loss(outputs, y_tuple, w_tuple)
            _sync_for_timing(device, in_profile_window)
            forward_end = time.perf_counter()

            if train:
                optimizer.zero_grad(set_to_none=True)
                backward_start = time.perf_counter()
                if grad_scaler is not None and grad_scaler.is_enabled():
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-4)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-4)
                    optimizer.step()
                _sync_for_timing(device, in_profile_window)
                backward_end = time.perf_counter()
            else:
                backward_start = forward_end
                backward_end = forward_end

            batch_size = int(x.shape[0])
            local_samples += batch_size
            total_loss += loss.item()
            n_steps += 1
            running_loss = total_loss / n_steps

            if in_profile_window:
                profile_steps += 1
                profile_samples += batch_size
                profile_data_time += data_wait
                profile_transfer_time += transfer_end - transfer_start
                profile_forward_time += forward_end - transfer_end
                profile_backward_time += backward_end - backward_start
                profile_step_time += backward_end - step_start

            if show_progress:
                batches.set_postfix(loss=f"{running_loss:.4f}")
            if log_every_batches and n_steps % log_every_batches == 0:
                message = (
                    f"{desc or 'epoch'} batch {n_steps}"
                    f" loss={loss.item():.4f} avg_loss={running_loss:.4f}"
                )
                if show_progress:
                    batches.write(message)
                else:
                    print(message)
            if wandb_log_every_batches and n_steps % wandb_log_every_batches == 0:
                prefix = wandb_prefix or ("train" if train else "val")
                metrics = {
                    f"{prefix}/batch_loss": loss.item(),
                    f"{prefix}/running_loss": running_loss,
                }
                if train:
                    metrics["optim/lr"] = optimizer.param_groups[0]["lr"]
                    metrics["optim/weight_decay"] = optimizer.param_groups[0][
                        "weight_decay"
                    ]
                log_wandb(
                    wandb_run,
                    metrics,
                    step=wandb_step_offset + n_steps,
                )
            data_start = time.perf_counter()

    epoch_time = time.perf_counter() - epoch_start
    metrics = _reduce_epoch_metrics(
        {
            "total_loss": total_loss,
            "n_steps": n_steps,
            "local_samples": local_samples,
            "epoch_time": epoch_time,
            "profile_samples": profile_samples,
            "profile_steps": profile_steps,
            "profile_data_time": profile_data_time,
            "profile_transfer_time": profile_transfer_time,
            "profile_forward_time": profile_forward_time,
            "profile_backward_time": profile_backward_time,
            "profile_step_time": profile_step_time,
        },
        device,
        world_size,
    )
    if return_metrics:
        return metrics["loss"], metrics
    return metrics["loss"]


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
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Global training batch size; divided evenly across DDP ranks",
    )
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--profile_batches", type=int, default=0)
    parser.add_argument("--profile_warmup_batches", type=int, default=10)
    parser.add_argument("--no_sync_batchnorm", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument(
        "--amp_dtype", choices=("none", "bf16", "fp16"), default="none"
    )
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        default=False,
        help="Resume from checkpoint_folder/latest_model.pt when it exists",
    )
    parser.add_argument("--no_progress", action="store_true", default=False)
    parser.add_argument("--log_every_batches", type=int, default=0)
    parser.add_argument("--wandb_log_every_batches", type=int, default=0)
    add_wandb_args(parser)
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
    per_rank_batch_size = resolve_per_rank_batch_size(args.batch_size, world_size)
    train_samples_per_rank = steps_per_epoch * per_rank_batch_size

    val_sw = tuple(k == 0 for k in range(num_species))
    val_dataset = SequenceDataset(
        val_dataset_files, args.input_length, args.output_length, val_sw
    )

    train_loader = build_weighted_dataloader(
        train_datasets,
        per_rank_batch_size,
        args.num_workers,
        rank,
        world_size,
        num_samples=train_samples_per_rank,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = build_weighted_dataloader(
        [val_dataset],
        per_rank_batch_size,
        args.num_workers,
        rank,
        world_size,
        shuffle=False,
        prefetch_factor=args.prefetch_factor,
    )

    model = build_model(args, output_dims, device, world_size, rank)
    amp_torch_dtype = resolve_amp_dtype(args.amp_dtype)
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp_dtype == "fp16" and device.type == "cuda"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    os.makedirs(args.checkpoint_folder, exist_ok=True)
    logger = CSVLogger(os.path.join(args.checkpoint_folder, "logs.csv"))
    start_epoch = 0
    best_val_loss = float("inf")
    resume_checkpoint = resolve_resume_checkpoint(args)
    if resume_checkpoint:
        start_epoch, best_val_loss, _ = load_training_checkpoint(
            model, optimizer, resume_checkpoint, device
        )
        if rank == 0:
            print(
                f"Resumed training from {resume_checkpoint} at epoch {start_epoch + 1}; best_val_loss={best_val_loss:.4f}"
            )

    args_dict = vars(args)
    args_dict["output_dims"] = output_dims
    args_dict["output_crop"] = args.input_length - args.output_length
    args_dict["dataset_sizes"] = dataset_sizes
    args_dict["start_epoch"] = start_epoch
    args_dict["global_batch_size"] = args.batch_size
    args_dict["per_rank_batch_size"] = per_rank_batch_size
    args_dict["world_size"] = world_size
    args_dict["amp_dtype"] = args.amp_dtype
    if rank == 0:
        gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
        print(
            "Training setup: "
            f"world_size={world_size} global_batch_size={args.batch_size} "
            f"per_rank_batch_size={per_rank_batch_size} num_workers={args.num_workers} "
            f"prefetch_factor={args.prefetch_factor} amp_dtype={args.amp_dtype} "
            f"sync_batchnorm={world_size > 1 and not args.no_sync_batchnorm} "
            f"compile={args.compile} gpu={gpu_name}",
            flush=True,
        )
    wandb_run = init_wandb(args, args_dict, rank=rank)

    for epoch in range(start_epoch, args.epochs):
        apply_optimizer_schedule(
            optimizer, args.learning_rate, args.weight_decay, args.epochs, epoch
        )
        lr_now = optimizer.param_groups[0]["lr"]
        wd_now = optimizer.param_groups[0]["weight_decay"]
        wandb_batch_log_every = args.wandb_log_every_batches
        if wandb_batch_log_every == 0 and args.log_every_batches > 0:
            wandb_batch_log_every = args.log_every_batches
        train_loss, train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            world_size=world_size,
            max_steps=steps_per_epoch,
            train=True,
            desc=f"train {epoch + 1}/{args.epochs}",
            show_progress=(rank == 0 and not args.no_progress),
            log_every_batches=args.log_every_batches if rank == 0 else 0,
            wandb_run=wandb_run if rank == 0 else None,
            wandb_prefix="train",
            wandb_step_offset=epoch * steps_per_epoch,
            wandb_log_every_batches=wandb_batch_log_every if rank == 0 else 0,
            amp_dtype=amp_torch_dtype,
            grad_scaler=grad_scaler,
            profile_batches=args.profile_batches,
            profile_warmup_batches=args.profile_warmup_batches,
            return_metrics=True,
        )
        if rank == 0:
            print(
                f"Train throughput: samples/sec={train_metrics['samples_per_sec']:.2f} "
                f"global_samples={train_metrics['global_samples']:.0f} "
                f"epoch_time={train_metrics['epoch_time_sec']:.2f}s",
                flush=True,
            )
            if args.profile_batches > 0:
                print(
                    f"Profile throughput: samples/sec={train_metrics['profile_samples_per_sec']:.2f} "
                    f"profile_steps={train_metrics['profile_steps']:.0f} "
                    f"data_wait={train_metrics['profile_data_wait_sec']:.4f}s "
                    f"transfer={train_metrics['profile_transfer_sec']:.4f}s "
                    f"forward_loss={train_metrics['profile_forward_loss_sec']:.4f}s "
                    f"backward_step={train_metrics['profile_backward_step_sec']:.4f}s",
                    flush=True,
                )
        if args.profile_batches > 0:
            if rank == 0:
                print("Profile run complete; skipping validation/checkpoint.", flush=True)
            break
        val_loss = _run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            world_size=world_size,
            train=False,
            desc=f"val {epoch + 1}/{args.epochs}",
            show_progress=(rank == 0 and not args.no_progress),
            amp_dtype=amp_torch_dtype,
        )

        if rank == 0:
            checkpoint_saved = val_loss < best_val_loss
            next_best_val_loss = min(best_val_loss, val_loss)
            print(
                f"Epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={lr_now:.2e}  wd={wd_now:.2e}"
            )
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr_now,
                "wd": wd_now,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "optim/lr": lr_now,
                "optim/weight_decay": wd_now,
                "best_val_loss": next_best_val_loss,
                "checkpoint_saved": int(checkpoint_saved),
                "train/samples_per_sec": train_metrics["samples_per_sec"],
                "train/global_samples": train_metrics["global_samples"],
                "train/epoch_time_sec": train_metrics["epoch_time_sec"],
            }
            logger.log(row)
            log_wandb(wandb_run, row, step=epoch + 1)

            if checkpoint_saved:
                best_val_loss = next_best_val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    None,
                    val_loss,
                    epoch,
                    args.checkpoint_folder,
                    args_dict,
                    checkpoint_name="best_model.pt",
                    best_val_loss=best_val_loss,
                )
            save_checkpoint(
                model,
                optimizer,
                None,
                val_loss,
                epoch,
                args.checkpoint_folder,
                args_dict,
                checkpoint_name="latest_model.pt",
                best_val_loss=best_val_loss,
            )

    finish_wandb(wandb_run)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
