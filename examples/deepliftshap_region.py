"""
Compute DeepLIFT/SHAP attribution scores for a genomic region and plot a sequence logo.

Usage:
    python examples/deepliftshap_region.py \\
        --model_checkpoint models/promoterAI_v1_hg38_mm10_finetune.pt \\
        [--fasta examples/data/hg38.fa] \\
        [--chrom chr12] [--start 131710989] [--end 131711189] \\
        [--n_shuffles 20] [--species mouse] [--tracks 0 5 7] [--compile] [--no-tf32] \\
        [--device cpu] \\
        [--output examples/img/deepliftshap.png]

The script extracts a full-length input window centered on the specified region,
runs DeepLIFT/SHAP (promoterai_torch.attribution.deep_lift_shap) on the mean
prediction of one species head (default the first; --species picks one by name from
the checkpoint's species_order) over its output window and the chosen tracks
(default all), and produces a two-panel figure. Like the promoterai-torch CLI, it
allows TF32 matmuls and convolutions on Ampere and newer GPUs (TensorFlow's default);
--no-tf32 runs strict float32. The figure has two panels:
  - Top: per-position contribution sum across the full input window
  - Bottom: sequence logo (contribution × one-hot) zoomed to the region of interest
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pyfaidx
import torch
from matplotlib import gridspec
from tangermeme.plot import plot_logo

from promoterai_torch.attribution import deep_lift_shap
from promoterai_torch.dataset import onehot_encode
from promoterai_torch.utils import add_tf32_argument, load_pretrained, set_tf32


def species_head(ckpt_args: dict, species: str | None) -> int:
    """Output-head index for a species name, from the checkpoint's species_order."""
    if species is None:
        return 0
    order = ckpt_args.get("species_order")
    if not order:
        raise ValueError(
            "this checkpoint doesn't record species_order (convert it with a current "
            "promoterai-torch convert), so --species can't be resolved"
        )
    if species not in order:
        raise ValueError(f"--species {species!r} not in the checkpoint's species_order {order}")
    return order.index(species)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model_checkpoint",
        required=True,
        help="Path to a .pt checkpoint from promoterai-torch convert or finetune",
    )
    parser.add_argument("--fasta", default="examples/data/hg38.fa")
    parser.add_argument("--chrom", default="chr12")
    parser.add_argument(
        "--start",
        type=int,
        default=131710989,
        help="0-based start of the region of interest",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=131711189,
        help="0-based end of the region of interest (exclusive)",
    )
    parser.add_argument("--input_length", type=int, default=20480)
    parser.add_argument("--n_shuffles", type=int, default=20)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument(
        "--species",
        default=None,
        help="species head to attribute, by name from the checkpoint's species_order "
        "(e.g. human, mouse; default: the first head)",
    )
    parser.add_argument(
        "--tracks", type=int, nargs="+", default=None,
        help="track indices into that head to attribute (default all)",
    )
    parser.add_argument(
        "--compile", action="store_true", help="fuse elementwise work (torch.compile / TorchScript)"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="examples/img/deepliftshap.png")
    add_tf32_argument(parser)
    args = parser.parse_args()
    set_tf32(args.tf32)

    # ── Extract sequence ──────────────────────────────────────────────────────
    center = (args.start + args.end) // 2
    half = args.input_length // 2
    seq_start = center - half
    seq_end = seq_start + args.input_length

    fa = pyfaidx.Fasta(args.fasta)
    chrom_len = len(fa[args.chrom])
    if seq_start < 0 or seq_end > chrom_len:
        raise ValueError(
            f"Input window {args.chrom}:{seq_start}-{seq_end} extends beyond chromosome "
            f"(length {chrom_len}). Choose a region further from the chromosome ends."
        )

    seq = str(fa[args.chrom][seq_start:seq_end]).upper()
    roi_s = args.start - seq_start  # ROI start within window
    roi_e = args.end - seq_start  # ROI end within window

    # ── Prepare input tensor: (1, 4, L) ──────────────────────────────────────
    x = torch.from_numpy(onehot_encode(seq)).float().T.unsqueeze(0)  # (1, 4, L)

    # ── Load model and move to device ─────────────────────────────────────────
    model, ckpt_args = load_pretrained(args.model_checkpoint)
    model = model.to(args.device)
    model.eval()
    head = species_head(ckpt_args, args.species)

    # ── Run DeepLIFT/SHAP, capturing convergence deltas ──────────────────────
    print(f"Running DeepLIFT/SHAP ({args.n_shuffles} shuffles) on {args.device}…")
    attrs, deltas = deep_lift_shap(
        model,
        x,
        head=head,
        tracks=args.tracks,
        n_shuffles=args.n_shuffles,
        random_state=args.random_state,
        compile=args.compile,
        return_deltas=True,
        device=args.device,
    )
    print(f"max convergence delta: {float(deltas.max()):.2e}")

    # Free GPU memory before plotting (attributions are returned on the CPU)
    if args.device != "cpu":
        model.cpu()
        torch.cuda.empty_cache()

    # Projected attributions: contribution of actual bases only
    contrib = (attrs * x).squeeze(0)  # (4, L) both on CPU
    contrib_sum = contrib.sum(dim=0)  # (L,) per-position sum

    # ── Plot ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    fig = plt.figure(figsize=(16, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.6)

    # Top: full-window contribution track
    ax_track = fig.add_subplot(gs[0])
    positions = np.arange(args.input_length)
    ax_track.fill_between(positions, contrib_sum.numpy(), alpha=0.7)
    ax_track.axvspan(roi_s, roi_e, color="red", alpha=0.15, label="ROI")
    ax_track.set_xlim(0, args.input_length)
    ax_track.set_xlabel("Position in input window (bp)")
    ax_track.set_ylabel("Contribution")
    ax_track.legend(fontsize=8, loc="upper right")
    ax_track.set_title(
        f"DeepLIFT/SHAP — {args.chrom}:{seq_start}–{seq_end} "
        f"(ROI: {args.chrom}:{args.start}–{args.end})",
        fontstyle="italic",
    )

    # Bottom: sequence logo over the ROI
    ax_logo = fig.add_subplot(gs[1])
    roi_attr = contrib[:, roi_s:roi_e]  # (4, roi_len)
    plot_logo(roi_attr, ax=ax_logo)
    roi_len = roi_e - roi_s
    tick_step = max(1, roi_len // 10)
    tick_pos = list(range(0, roi_len, tick_step))
    ax_logo.set_xticks(tick_pos)
    ax_logo.set_xticklabels(
        [str(args.start + p) for p in tick_pos], rotation=45, ha="right"
    )
    ax_logo.set_xlabel(f"Genomic position ({args.chrom})")
    ax_logo.set_ylabel("Contribution score")
    ax_logo.set_title(f"ROI: {args.chrom}:{args.start}–{args.end}", fontstyle="italic")

    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
