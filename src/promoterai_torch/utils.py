from __future__ import annotations

import csv
import os
from typing import Callable

import torch
import torch.nn as nn


def make_lr_lambda(total_epochs: int) -> Callable[[int], float]:
    """Return a lr_lambda for LambdaLR: linear warmup 0–10%, constant 10–90%, linear decay 90–100%."""

    def lr_lambda(epoch: int) -> float:
        if epoch < 0.1 * total_epochs:
            return (epoch + 1) / (0.1 * total_epochs)
        elif epoch > 0.9 * total_epochs:
            return (total_epochs - epoch) / (0.1 * total_epochs)
        else:
            return 1.0

    return lr_lambda


def apply_optimizer_schedule(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    base_wd: float,
    total_epochs: int,
    epoch: int,
) -> float:
    """Apply PromoterAI's epoch-begin LR/WD schedule and return the scale."""
    scale = make_lr_lambda(total_epochs)(epoch)
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * scale
        pg["weight_decay"] = base_wd * scale
    return scale


class WeightDecayScheduler:
    """Mirrors the LR schedule for weight_decay (LambdaLR does not touch it)."""

    def __init__(
        self, optimizer: torch.optim.Optimizer, base_wd: float, total_epochs: int
    ):
        """Store optimizer and build the same scale function used by make_lr_lambda."""
        self.optimizer = optimizer
        self.base_wd = base_wd
        self.scale_fn = make_lr_lambda(total_epochs)

    def step(self, epoch: int):
        """Scale weight_decay for all param groups by the triangle schedule factor."""
        scale = self.scale_fn(epoch)
        for pg in self.optimizer.param_groups:
            pg["weight_decay"] = self.base_wd * scale


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model from DDP and torch.compile wrappers."""
    base = model
    while hasattr(base, "module"):
        base = base.module
    while hasattr(base, "_orig_mod"):
        base = base._orig_mod
    return base


def normalize_model_state_dict(state_dict: dict) -> dict:
    """Strip common wrapper prefixes from checkpoint state_dict keys."""
    normalized = {}
    for key, value in state_dict.items():
        clean_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
                    changed = True
        normalized[clean_key] = value
    return normalized


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    val_loss: float,
    epoch: int,
    checkpoint_folder: str,
    args_dict: dict,
    checkpoint_name: str = "best_model.pt",
    best_val_loss: float | None = None,
):
    """Save model (unwrapped from DDP), optimizer, scheduler, and training metadata."""
    base = unwrap_model(model)
    os.makedirs(checkpoint_folder, exist_ok=True)
    torch.save(
        {
            "model_state_dict": base.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "val_loss": val_loss,
            "best_val_loss": val_loss if best_val_loss is None else best_val_loss,
            "epoch": epoch,
            "args": args_dict,
        },
        os.path.join(checkpoint_folder, checkpoint_name),
    )


def load_pretrained(checkpoint_path: str, map_location: str = "cpu"):
    """Load a checkpoint and reconstruct PromoterAI. Returns (model, args_dict)."""
    from promoterai_torch.architecture import PromoterAI

    ckpt = torch.load(checkpoint_path, map_location=map_location)
    args = ckpt["args"]
    model = PromoterAI(
        num_blocks=args["num_blocks"],
        model_dim=args["model_dim"],
        output_dims=args["output_dims"],
        output_crop=args["output_crop"]
        if "output_crop" in args
        else args["input_length"] - args["output_length"],
        shortcut_layer_freq=args.get("shortcut_layer_freq", 4),
    )
    model.load_state_dict(normalize_model_state_dict(ckpt["model_state_dict"]))
    return model, args


class CSVLogger:
    def __init__(self, path: str):
        """Open (or append to) a CSV log file at path; writes header on first call to log()."""
        self.path = path
        self._header_written = os.path.exists(path)

    def log(self, row: dict):
        """Append a row dict to the CSV, writing a header if this is the first call."""
        write_header = not self._header_written
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


def add_wandb_args(parser):
    """Add optional Weights & Biases logging arguments to a training parser."""
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument(
        "--wandb_mode", choices=("online", "offline", "disabled"), default=None
    )
    parser.add_argument("--wandb_tags", nargs="*", default=[])


def init_wandb(args, config: dict, rank: int = 0):
    """Initialize wandb on rank 0 when --wandb_project is provided."""
    if rank != 0:
        return None
    if getattr(args, "wandb_mode", None) == "disabled":
        return None
    project = getattr(args, "wandb_project", None)
    if not project:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb logging requires the optional wandb dependency. "
            'Install with: pip install "promoterai-torch[wandb]"'
        ) from exc

    init_kwargs = {
        "project": project,
        "entity": getattr(args, "wandb_entity", None),
        "name": getattr(args, "wandb_run_name", None),
        "mode": getattr(args, "wandb_mode", None),
        "tags": getattr(args, "wandb_tags", None) or None,
        "config": config,
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    return wandb.init(**init_kwargs)


def log_wandb(run, metrics: dict, step: int | None = None):
    """Log metrics to an optional wandb run."""
    if run is not None:
        run.log(metrics, step=step)


def finish_wandb(run):
    """Finish an optional wandb run."""
    if run is not None:
        run.finish()


def convert_tf_weights(
    keras_model_path: str,
    output_pt_path: str,
    input_length: int | None = None,
    output_length: int | None = None,
):
    """
    Convert a pretrained PromoterAI Keras SavedModel to a PyTorch checkpoint.

    Architecture (num_blocks, model_dim, output_dims, shortcut_layer_freq) is inferred
    from weight names. output_crop is derived from input_length - output_length when
    both are provided. input_length and output_length are stored as metadata.

    Expected weight naming (subclassed model format used by Illumina):
      dense/kernel, dense/bias                          — stem
      meta_former_block[_N]/depthwise_conv1d/...        — block dw conv
      meta_former_block[_N]/batch_normalization[_1]/... — block BN (bn1, bn2)
      meta_former_block[_N]/dense[_1]/...               — block FFN
      shortcut_{species}{block_num}/kernel|bias         — output head projections
    """
    import re

    try:
        import tf_keras
    except ImportError:
        raise ImportError(
            "tensorflow-cpu and tf-keras are required for weight conversion. "
            "Install with: pip install tensorflow-cpu tf-keras"
        )

    from promoterai_torch.architecture import PromoterAI

    print(f"Loading Keras model from {keras_model_path} ...")
    keras_model = tf_keras.models.load_model(keras_model_path)

    # Flat weight dict: strip trailing ':0' from TF variable names
    w = {v.name.rstrip(":0"): v.numpy() for v in keras_model.weights}

    # ── Infer architecture ────────────────────────────────────────────────────
    if "dense/kernel" not in w:
        raise ValueError(
            "No 'dense/kernel' stem weight found. Is this a PromoterAI subclassed model?"
        )
    model_dim = w["dense/bias"].shape[0]

    block_idx_set: set[int] = set()
    for name in w:
        m = re.match(r"^meta_former_block(?:_(\d+))?/", name)
        if m:
            block_idx_set.add(int(m.group(1)) if m.group(1) else 0)
    if not block_idx_set:
        raise ValueError(
            "No MetaFormerBlock weights found — is this a PromoterAI model?"
        )
    num_blocks = max(block_idx_set) + 1

    # Collect shortcut projections.
    # Multi-species models: shortcut_{species}{N}/kernel  (e.g. shortcut_human24/kernel)
    # Single-species models: shortcut{N}/kernel           (e.g. shortcut24/kernel)
    shortcut_pat = re.compile(r"^shortcut(?:_([a-zA-Z]+))?(\d+)/kernel$")
    species_order: list[str] = []
    species_blocks: dict[str, list[int]] = {}
    for key in w:  # w is already stripped; preserves insertion order (Python 3.7+)
        m = shortcut_pat.match(key)
        if m:
            species = m.group(1) or ""  # empty string for single-species models
            block_num = int(m.group(2))
            if species not in species_order:
                species_order.append(species)
                species_blocks[species] = []
            if block_num not in species_blocks[species]:
                species_blocks[species].append(block_num)
    if not species_order:
        raise ValueError("No shortcut_{species}{N} output head weights found.")

    def _shortcut_key(species, block_num):
        return (
            f"shortcut_{species}{block_num}/kernel"
            if species
            else f"shortcut{block_num}/kernel"
        )

    output_dims = [
        w[_shortcut_key(s, species_blocks[s][0])].shape[1] for s in species_order
    ]

    shortcut_nums = sorted(species_blocks[species_order[0]])  # ascending
    shortcut_layer_freq = (
        shortcut_nums[1] - shortcut_nums[0]
        if len(shortcut_nums) > 1
        else shortcut_nums[0]
    )
    output_crop = (
        (input_length - output_length) if (input_length and output_length) else 0
    )

    print(
        f"Inferred: num_blocks={num_blocks}, model_dim={model_dim}, "
        f"output_dims={output_dims}, shortcut_layer_freq={shortcut_layer_freq}, "
        f"output_crop={output_crop}"
    )

    # ── Build PyTorch model ───────────────────────────────────────────────────
    pt_model = PromoterAI(
        num_blocks=num_blocks,
        model_dim=model_dim,
        output_dims=output_dims,
        shortcut_layer_freq=shortcut_layer_freq,
        output_crop=output_crop,
    )
    new_sd: dict[str, torch.Tensor] = {}

    # ── Stem: Dense (4, model_dim) → Conv1d (model_dim, 4, 1) ────────────────
    new_sd["stem.weight"] = torch.from_numpy(w["dense/kernel"].T[:, :, None])
    new_sd["stem.bias"] = torch.from_numpy(w["dense/bias"])

    # ── MetaFormer blocks ─────────────────────────────────────────────────────
    for i in range(num_blocks):
        kp = "meta_former_block" if i == 0 else f"meta_former_block_{i}"
        bp = f"blocks.{i}"

        def _bn(pt_pfx, keras_bn):
            new_sd[f"{pt_pfx}.weight"] = torch.from_numpy(w[f"{kp}/{keras_bn}/gamma"])
            new_sd[f"{pt_pfx}.bias"] = torch.from_numpy(w[f"{kp}/{keras_bn}/beta"])
            new_sd[f"{pt_pfx}.running_mean"] = torch.from_numpy(
                w[f"{kp}/{keras_bn}/moving_mean"]
            )
            new_sd[f"{pt_pfx}.running_var"] = torch.from_numpy(
                w[f"{kp}/{keras_bn}/moving_variance"]
            )
            new_sd[f"{pt_pfx}.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)

        _bn(f"{bp}.bn1", "batch_normalization")

        # DepthwiseConv1D kernel: (k, C, 1) → PT grouped conv (C, 1, k)
        kernel = w[f"{kp}/depthwise_conv1d/depthwise_kernel"]
        new_sd[f"{bp}.dw_conv.weight"] = torch.from_numpy(kernel.transpose(1, 2, 0))
        new_sd[f"{bp}.dw_conv.bias"] = torch.from_numpy(
            w[f"{kp}/depthwise_conv1d/bias"]
        )

        _bn(f"{bp}.bn2", "batch_normalization_1")

        # FFN Dense (in, out) → Linear weight (out, in)
        new_sd[f"{bp}.ffn1.weight"] = torch.from_numpy(w[f"{kp}/dense/kernel"].T)
        new_sd[f"{bp}.ffn1.bias"] = torch.from_numpy(w[f"{kp}/dense/bias"])
        new_sd[f"{bp}.ffn2.weight"] = torch.from_numpy(w[f"{kp}/dense_1/kernel"].T)
        new_sd[f"{bp}.ffn2.bias"] = torch.from_numpy(w[f"{kp}/dense_1/bias"])

    # ── Output heads ──────────────────────────────────────────────────────────
    # Projections are ordered highest block → lowest (matching shortcut_indices)
    shortcut_nums_desc = sorted(shortcut_nums, reverse=True)
    for j, species in enumerate(species_order):
        for p_idx, block_num in enumerate(shortcut_nums_desc):
            pfx = (
                f"shortcut_{species}{block_num}" if species else f"shortcut{block_num}"
            )
            new_sd[f"output_heads.{j}.projections.{p_idx}.weight"] = torch.from_numpy(
                w[f"{pfx}/kernel"].T
            )
            new_sd[f"output_heads.{j}.projections.{p_idx}.bias"] = torch.from_numpy(
                w[f"{pfx}/bias"]
            )

    # ── Load and verify ───────────────────────────────────────────────────────
    missing = set(pt_model.state_dict()) - set(new_sd)
    extra = set(new_sd) - set(pt_model.state_dict())
    if extra:
        print(
            f"  Warning: {len(extra)} unexpected keys ignored: {sorted(extra)[:3]}..."
        )
    if missing:
        print(
            f"  Warning: {len(missing)} keys not converted (will use random init): {sorted(missing)[:3]}..."
        )

    pt_model.load_state_dict(new_sd, strict=False)

    args_dict: dict = {
        "num_blocks": num_blocks,
        "model_dim": model_dim,
        "output_dims": output_dims,
        "output_crop": output_crop,
        "shortcut_layer_freq": shortcut_layer_freq,
    }
    if input_length is not None:
        args_dict["input_length"] = input_length
    if output_length is not None:
        args_dict["output_length"] = output_length

    torch.save(
        {"model_state_dict": pt_model.state_dict(), "args": args_dict}, output_pt_path
    )
    n_converted = len(new_sd) - len(missing)
    print(
        f"Converted {n_converted}/{len(pt_model.state_dict())} tensors → {output_pt_path}"
    )
