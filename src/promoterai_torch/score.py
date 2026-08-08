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
from torch.utils.data import DataLoader

from promoterai_torch.architecture import TwinModel
from promoterai_torch.dataset import VariantDataset
from promoterai_torch.utils import load_pretrained


def _collate_variant(batch):
    """Stack ref/alt tensors and labels from a list of VariantDataset items into batched tensors."""
    x_refs = torch.stack([item[0][0] for item in batch])
    x_alts = torch.stack([item[0][1] for item in batch])
    ys = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    return (x_refs, x_alts), ys


def main():
    """Score variants and write tanh-scaled scores to a TSV alongside the input columns."""
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
        required=True,
        help="Input sequence length in bp (must match the model checkpoint)",
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
        "--compile",
        action="store_true",
        default=False,
        help="torch.compile the model; pays off on large variant files, not small ones",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Show a tqdm progress bar"
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model, _ = load_pretrained(args.model_checkpoint, map_location=str(device))
    twin_model = TwinModel(base_model).to(device)
    twin_model.eval()
    if args.compile:
        twin_model = torch.compile(twin_model)

    df_var = pd.read_csv(args.var_file, sep="\t")
    dataset = VariantDataset(df_var, args.fasta_file, args.input_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_variant,
    )

    all_diffs = []
    with torch.no_grad():
        from tqdm import tqdm

        batches = tqdm(
            loader, desc="Scoring variants", unit="batch", disable=not args.verbose
        )
        for (x_ref, x_alt), _ in batches:
            diff = twin_model(x_ref.to(device), x_alt.to(device))
            all_diffs.append(diff.cpu().numpy())

    scores = np.tanh(np.concatenate(all_diffs).round(4))
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
