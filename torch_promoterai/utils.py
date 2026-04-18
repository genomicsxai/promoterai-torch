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


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    val_loss: float,
    epoch: int,
    checkpoint_folder: str,
    args_dict: dict,
):
    """Save model (unwrapped from DDP), optimizer, and scheduler to best_model.pt."""
    base = model.module if hasattr(model, "module") else model
    os.makedirs(checkpoint_folder, exist_ok=True)
    torch.save(
        {
            "model_state_dict": base.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "epoch": epoch,
            "args": args_dict,
        },
        os.path.join(checkpoint_folder, "best_model.pt"),
    )


def load_pretrained(checkpoint_path: str, map_location: str = "cpu"):
    """Load a checkpoint and reconstruct PromoterAI. Returns (model, args_dict)."""
    from torch_promoterai.architecture import PromoterAI

    ckpt = torch.load(checkpoint_path, map_location=map_location)
    args = ckpt["args"]
    model = PromoterAI(
        num_blocks=args["num_blocks"],
        model_dim=args["model_dim"],
        output_dims=args["output_dims"],
        output_crop=args["output_crop"]
        if "output_crop" in args
        else args["input_length"] - args["output_length"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
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


def convert_tf_weights(
    keras_model_path: str,
    output_pt_path: str,
    input_length: int | None = None,
    output_length: int | None = None,
):
    """
    Convert a pretrained PromoterAI Keras SavedModel to a PyTorch checkpoint.

    Architecture parameters (num_blocks, model_dim, output_dims, output_crop) are
    inferred automatically from the Keras model. input_length and output_length are
    optional metadata stored in the checkpoint for convenience.

    Requires tensorflow: pip install 'torch-promoterai[convert]'
    """
    try:
        import tf_keras
    except ImportError:
        raise ImportError(
            "tensorflow-cpu and tf-keras are required for weight conversion. "
            "Install with: pip install tensorflow-cpu tf-keras"
        )

    from torch_promoterai.architecture import PromoterAI

    print(f"Loading Keras model from {keras_model_path} ...")
    keras_model = tf_keras.models.load_model(keras_model_path)

    # ── Infer architecture from layer class names (type-agnostic across TF versions) ──
    def _cls(lay) -> str:
        return type(lay).__name__

    conv_layers = [lay for lay in keras_model.layers if _cls(lay) == "Conv1D"]
    bn_layers = [lay for lay in keras_model.layers if _cls(lay) == "BatchNormalization"]
    dw_layers = [lay for lay in keras_model.layers if _cls(lay) == "DepthwiseConv1D"]
    all_dense = [lay for lay in keras_model.layers if _cls(lay) == "Dense"]
    crop_layers = [lay for lay in keras_model.layers if _cls(lay) == "Cropping1D"]

    # Dense layers whose names start with 'output{j}_' are output head projections
    output_dense = [
        lay for lay in all_dense if lay.name.split("_")[0].startswith("output")
    ]
    ffn_dense = [lay for lay in all_dense if lay not in output_dense]

    if not conv_layers:
        raise ValueError("Could not find Conv1D stem layer in the Keras model.")
    if not dw_layers:
        raise ValueError(
            "Could not find DepthwiseConv1D layers — is this a PromoterAI model?"
        )

    num_blocks = len(dw_layers)
    model_dim = conv_layers[0].filters

    # Group output head Denses by head index (j in 'output{j}_{layer_idx}')
    import re

    head_groups: dict[int, list] = {}
    for layer in output_dense:
        m = re.match(r"^output(\d+)_", layer.name)
        if m:
            j = int(m.group(1))
            head_groups.setdefault(j, []).append(layer)
    if not head_groups:
        raise ValueError(
            "Could not find output head Dense layers named 'output{j}_{i}'."
        )

    output_dims = [head_groups[j][0].units for j in sorted(head_groups)]
    output_crop = crop_layers[0].cropping[0] * 2 if crop_layers else 0

    print(
        f"Inferred: num_blocks={num_blocks}, model_dim={model_dim}, "
        f"output_dims={output_dims}, output_crop={output_crop}"
    )

    # ── Build PyTorch model ───────────────────────────────────────────────────
    pt_model = PromoterAI(
        num_blocks=num_blocks,
        model_dim=model_dim,
        output_dims=output_dims,
        output_crop=output_crop,
    )
    new_sd: dict[str, torch.Tensor] = {}

    def _bn(keras_layer, pt_prefix: str):
        w = {v.name.split("/")[-1].rstrip(":0"): v.numpy() for v in keras_layer.weights}
        mapping = {
            "gamma": "weight",
            "beta": "bias",
            "moving_mean": "running_mean",
            "moving_variance": "running_var",
        }
        for k_name, pt_name in mapping.items():
            if k_name in w:
                new_sd[f"{pt_prefix}.{pt_name}"] = torch.from_numpy(w[k_name])
        new_sd[f"{pt_prefix}.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)

    def _conv1d(keras_layer, pt_prefix: str):
        # TF Conv1D kernel: (kernel_size, in_ch, out_ch) → PT: (out_ch, in_ch, kernel_size)
        w = {v.name.split("/")[-1].rstrip(":0"): v.numpy() for v in keras_layer.weights}
        new_sd[f"{pt_prefix}.weight"] = torch.from_numpy(w["kernel"].transpose(2, 1, 0))
        new_sd[f"{pt_prefix}.bias"] = torch.from_numpy(w["bias"])

    def _dw_conv(keras_layer, pt_prefix: str):
        # TF DepthwiseConv1D kernel: (kernel_size, in_ch, depth_mult) → PT: (in_ch, depth_mult, kernel_size)
        # For groups conv: PT weight shape is (out_ch=in_ch, in_ch/groups=1, kernel_size)
        w = {v.name.split("/")[-1].rstrip(":0"): v.numpy() for v in keras_layer.weights}
        # depthwise_kernel shape: (kernel_size, in_channels, depth_multiplier)
        kernel = w["depthwise_kernel"]  # (k, C, 1)
        new_sd[f"{pt_prefix}.weight"] = torch.from_numpy(
            kernel.transpose(1, 2, 0)
        )  # (C, 1, k)
        if "bias" in w:
            new_sd[f"{pt_prefix}.bias"] = torch.from_numpy(w["bias"])

    def _dense(keras_layer, pt_prefix: str):
        # TF Dense kernel: (in, out) → PT Linear: (out, in)
        w = {v.name.split("/")[-1].rstrip(":0"): v.numpy() for v in keras_layer.weights}
        new_sd[f"{pt_prefix}.weight"] = torch.from_numpy(w["kernel"].T)
        new_sd[f"{pt_prefix}.bias"] = torch.from_numpy(w["bias"])

    # ── Stem ─────────────────────────────────────────────────────────────────
    _conv1d(conv_layers[0], "stem")

    # ── MetaFormer blocks ─────────────────────────────────────────────────────
    # Layer order per block: bn1, dw_conv, bn2, ffn1, ffn2
    if len(bn_layers) != num_blocks * 2:
        raise ValueError(f"Expected {num_blocks * 2} BN layers, found {len(bn_layers)}")
    if len(ffn_dense) != num_blocks * 2:
        raise ValueError(
            f"Expected {num_blocks * 2} FFN Dense layers, found {len(ffn_dense)}"
        )

    for i in range(num_blocks):
        bp = f"blocks.{i}"
        _bn(bn_layers[2 * i], f"{bp}.bn1")
        _dw_conv(dw_layers[i], f"{bp}.dw_conv")
        _bn(bn_layers[2 * i + 1], f"{bp}.bn2")
        _dense(ffn_dense[2 * i], f"{bp}.ffn1")
        _dense(ffn_dense[2 * i + 1], f"{bp}.ffn2")

    # ── Output heads ──────────────────────────────────────────────────────────
    shortcut_indices = list(range(num_blocks, 0, -4))
    for j in sorted(head_groups):
        # Sort head projection layers by the layer index in their name (descending, matching shortcut_indices)
        head_layers = sorted(
            head_groups[j],
            key=lambda lay: int(re.search(r"_(\d+)$", lay.name).group(1)),
            reverse=True,
        )
        for p_idx, (layer, _) in enumerate(zip(head_layers, shortcut_indices)):
            _dense(layer, f"output_heads.{j}.projections.{p_idx}")

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

    args_dict = {
        "num_blocks": num_blocks,
        "model_dim": model_dim,
        "output_dims": output_dims,
        "output_crop": output_crop,
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
