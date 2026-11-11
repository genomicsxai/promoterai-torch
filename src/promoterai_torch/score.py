"""
Variant effect scoring CLI.

Usage:
    promoterai-torch \
        --model_checkpoint <path/to/best_model.pt> \
        --var_file variants.tsv \
        --fasta_file <genome.fa> \
        --input_length 20480
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from promoterai_torch.architecture import TwinModel
from promoterai_torch.dataset import VariantDataset, collate_variant
from promoterai_torch.incremental import edit_span, reference_layers, substitution_diffs
from promoterai_torch.utils import (
    add_tf32_argument,
    load_pretrained,
    resolve_input_length,
    set_tf32,
)

# Alts recomputed together on one cached reference (a site has at most 3 SNVs;
# MNV sites can have more).
_ALT_CHUNK = 16


def _site_groups(df_var: pd.DataFrame) -> tuple[list[list[int]], np.ndarray]:
    """Split variants into same-length substitutions grouped by site, and the rest.

    Returns (groups, other): groups is a list of positional-index lists, one per
    (chrom, pos, strand) site, of variants with len(ref) == len(alt), which can
    share one reference forward pass; other holds the positional indices of
    length-changing variants (indels), which need full passes.
    """
    same_length = (
        df_var["ref"].astype(str).str.len() == df_var["alt"].astype(str).str.len()
    ).to_numpy()
    strand = df_var["strand"] if "strand" in df_var.columns else [1] * len(df_var)
    keys = list(zip(df_var["chrom"], df_var["pos"], strand))
    groups: dict = {}
    for i in np.flatnonzero(same_length):
        groups.setdefault(keys[i], []).append(int(i))
    return list(groups.values()), np.flatnonzero(~same_length)


class _SiteDataset(Dataset):
    """Each item is one site's variants, as [(x_ref, x_alts, indices), ...].

    A site's variants normally share one reference window; a variant whose
    reference allele doesn't match (an all-zero placeholder) or that otherwise
    has a different x_ref gets its own entry.
    """

    def __init__(self, variants: VariantDataset, groups: list[list[int]]):
        self.variants = variants
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, g: int):
        entries: list[tuple[torch.Tensor, list, list]] = []
        for i in self.groups[g]:
            (x_ref, x_alt), _ = self.variants[i]
            for ref, alts, idx in entries:
                if torch.equal(ref, x_ref):
                    alts.append(x_alt)
                    idx.append(i)
                    break
            else:
                entries.append((x_ref, [x_alt], [i]))
        return [(ref, torch.stack(alts), idx) for ref, alts, idx in entries]


def _identity(item):
    return item


@torch.no_grad()
def _score_sites(model, sites: DataLoader, device, diffs: np.ndarray, verbose: bool) -> None:
    """Fill diffs[i] for every grouped substitution, one cached reference per site."""
    from tqdm import tqdm

    for entries in tqdm(sites, desc="Scoring sites", unit="site", disable=not verbose):
        for x_ref, x_alts, idx in entries:
            x_ref = x_ref.unsqueeze(0).to(device)
            x_alts = x_alts.to(device)
            span = edit_span(x_ref, x_alts)
            if span is None:  # placeholders or alt == ref: TwinModel's diff is exactly 0
                diffs[idx] = 0.0
                continue
            layers = reference_layers(model, x_ref)
            for k in range(0, len(idx), _ALT_CHUNK):
                d = substitution_diffs(model, layers, x_alts[k : k + _ALT_CHUNK], span)
                diffs[idx[k : k + _ALT_CHUNK]] = d.cpu().numpy()
            del layers


@torch.no_grad()
def variant_diffs(
    base_model,
    df_var: pd.DataFrame,
    fasta_file,
    input_length: int,
    *,
    device,
    batch_size: int = 2,
    num_workers: int = 0,
    incremental: bool = True,
    twin_model=None,
    verbose: bool = False,
) -> np.ndarray:
    """TwinModel diffs mean(head0(alt) - head0(ref)) for every row of df_var, float32.

    With `incremental`, same-length substitutions are grouped by site and scored
    from one cached reference pass each (`incremental.substitution_diffs`); the
    rest (indels, or everything without `incremental`) use full ref/alt passes
    through `twin_model` (default: ``TwinModel(base_model)``). Callers apply
    ``np.tanh(diffs.round(4))`` for scores.
    """
    device = torch.device(device)
    if twin_model is None:
        twin_model = TwinModel(base_model).to(device).eval()
    dataset = VariantDataset(df_var, str(fasta_file), input_length)

    # float32, like the model's outputs, so rounding matches the full-pass path.
    diffs = np.full(len(df_var), np.nan, dtype=np.float32)
    if incremental:
        groups, full_idx = _site_groups(df_var)
    else:
        groups, full_idx = [], np.arange(len(df_var))

    if groups:
        sites = DataLoader(
            _SiteDataset(dataset, groups),
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_identity,
        )
        _score_sites(base_model, sites, device, diffs, verbose)

    if len(full_idx):
        loader = DataLoader(
            Subset(dataset, full_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_variant,
        )
        from tqdm import tqdm

        full_diffs = []
        for (x_ref, x_alt), _ in tqdm(
            loader, desc="Scoring variants", unit="batch", disable=not verbose
        ):
            full_diffs.append(twin_model(x_ref.to(device), x_alt.to(device)).cpu().numpy())
        diffs[full_idx] = np.concatenate(full_diffs)
    return diffs


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Build (or populate, when composed into the unified CLI) the score parser."""
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_checkpoint",
        required=True,
        help="Path to a trained/converted PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--var_file",
        required=True,
        help="TSV of variants to score, with chrom/pos/ref/alt/strand columns",
    )
    parser.add_argument(
        "--fasta_file", required=True, help="Reference genome FASTA (indexed with pyfaidx)"
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=None,
        help=(
            "Input sequence length in bp (must match the model checkpoint); "
            "falls back to the checkpoint's stored input_length, then 20480 "
            "(the published PromoterAI model's length), if omitted"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Number of variants scored per forward pass (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to score on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes for sequence extraction (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: <var_file_stem>.<model_stem><ext>)",
    )
    parser.add_argument(
        "--incremental",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score same-length substitutions (SNVs/MNVs; on by default) by running each site's "
        "reference once and recomputing only the receptive-field window each alt "
        "can change (exact up to float rounding; ~4.5x faster on SNV sets with "
        "all three alts per site). Indels always use full passes. "
        "--no-incremental scores everything with full ref/alt passes",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="torch.compile the model for full ref/alt passes (indels, or everything "
        "with --no-incremental); pays off on large variant files, not small ones",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Show a tqdm progress bar"
    )
    add_tf32_argument(parser)
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    """Score variants (parsing args if not already parsed) and write tanh-scaled scores to a TSV."""
    if args is None:
        args = build_parser().parse_args()
    set_tf32(getattr(args, "tf32", True))

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model, base_args = load_pretrained(args.model_checkpoint, map_location=str(device))
    twin_model = TwinModel(base_model).to(device)
    twin_model.eval()
    if args.compile:
        twin_model = torch.compile(twin_model)

    args.input_length = resolve_input_length(args.input_length, base_args)
    df_var = pd.read_csv(args.var_file, sep="\t")
    diffs = variant_diffs(
        base_model,
        df_var,
        args.fasta_file,
        args.input_length,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        incremental=args.incremental,
        twin_model=twin_model,
        verbose=args.verbose,
    )

    scores = np.tanh(diffs.round(4))
    df_var["score"] = scores

    if args.output:
        out_path = Path(args.output)
    else:
        var_path = Path(args.var_file)
        model_name = Path(args.model_checkpoint).stem
        out_path = var_path.parent / f"{var_path.stem}.{model_name}{var_path.suffix}"
    df_var.to_csv(out_path, sep="\t", index=False)
    print(f"Scores written to {out_path}")


if __name__ == "__main__":
    main()
