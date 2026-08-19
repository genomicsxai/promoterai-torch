"""
Compute DeepLIFT/SHAP attribution scores for a genomic region and plot a sequence logo.

Usage:
    python examples/deepliftshap_region.py \\
        --model_checkpoint models/promoterAI_v1_hg38_mm10_finetune.pt \\
        [--fasta examples/data/hg38.fa] \\
        [--chrom chr12] [--start 131710989] [--end 131711189] \\
        [--n_shuffles 20] [--device cpu] [--output examples/img/deepliftshap.png]

The script extracts a full-length input window centered on the specified region,
runs DeepLIFT/SHAP, and produces a two-panel figure:
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
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.plot import plot_logo
from torch import nn

from promoterai_torch.dataset import onehot_encode
from promoterai_torch.utils import load_pretrained


class _Wrapper(nn.Module):
    """Adapt PromoterAI for tangermeme: (B, 4, L) in → (B, n_tracks) out."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # (B, 1) — mean over positions and tracks
        out = self.model(x.transpose(1, 2))[0].mean(dim=(1, 2)).unsqueeze(1)
        return out


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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="examples/img/deepliftshap.png")
    args = parser.parse_args()

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
    model, _ = load_pretrained(args.model_checkpoint)
    model = model.to(args.device)
    model.eval()
    wrapper = _Wrapper(model)

    # ── Predict to get output delta ─────────────────────────────────────────

    # ── Run DeepLIFT/SHAP, capturing convergence deltas ──────────────────────
    print(f"Running DeepLIFT/SHAP ({args.n_shuffles} shuffles) on {args.device}…")
    attrs = deep_lift_shap(
        wrapper,
        x,
        n_shuffles=args.n_shuffles,
        batch_size=args.batch_size,
        device=args.device,
        print_convergence_deltas=True,
    )

    # Move results to CPU and free GPU memory before plotting
    attrs = attrs.cpu()
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
