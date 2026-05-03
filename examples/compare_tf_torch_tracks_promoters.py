#!/usr/bin/env python
"""
Compare TF/Keras and PyTorch PromoterAI predicted tracks on genomic promoters.

Promoters can be supplied explicitly as NAME:CHROM:TSS:STRAND, where TSS is a
1-based genomic coordinate and STRAND is +, -, 1, or -1. They can also be read
from TSV files with columns chrom, tss_pos, strand, and optionally gene.

Example:
    python examples/compare_tf_torch_tracks_promoters.py \
        --keras_model models/promoterAI_v1_hg38_mm10_finetune \
        --torch_checkpoint models/promoterAI_v1_hg38_mm10_finetune.pt \
        --fasta hg38.fa \
        --promoter_tsv examples/data/promoterAI_tss500_TERT.scores.tsv.gz \
        --promoter_tsv examples/data/promoterAI_tss500_SFSWAP.scores.tsv.gz \
        --promoter_tsv examples/data/promoterAI_tss500_DNAJC9.scores.tsv.gz
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyfaidx
import torch
import tqdm

from promoterai_torch.dataset import onehot_encode
from promoterai_torch.utils import load_pretrained
from track_parity_utils import (
    clear_tf_runtime,
    compare_track_outputs,
    configure_tf_runtime,
    load_tf_model,
    predict_tf,
    predict_torch,
    print_results,
    write_results_csv,
)


@dataclass(frozen=True)
class Promoter:
    name: str
    chrom: str
    tss: int
    strand: str


def parse_promoter(spec: str) -> Promoter:
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"Promoter spec must be NAME:CHROM:TSS:STRAND, got {spec!r}"
        )
    name, chrom, tss, strand = parts
    return Promoter(name=name, chrom=chrom, tss=int(tss), strand=strand)


def promoters_from_tsv(path: str | Path, limit: int | None = None) -> list[Promoter]:
    df = pd.read_csv(path, sep="\t")
    required = {"chrom", "tss_pos", "strand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    name_col = "gene" if "gene" in df.columns else None
    cols = ["chrom", "tss_pos", "strand"] + ([name_col] if name_col else [])
    unique = df[cols].drop_duplicates()
    if limit is not None:
        unique = unique.head(limit)

    promoters = []
    stem = Path(path).stem.replace(".scores", "")
    for i, row in unique.reset_index(drop=True).iterrows():
        name = str(row[name_col]) if name_col else f"{stem}_{i}"
        promoters.append(
            Promoter(
                name=name,
                chrom=str(row["chrom"]),
                tss=int(row["tss_pos"]),
                strand=str(row["strand"]),
            )
        )
    return promoters


def sequence_for_promoter(
    fasta: pyfaidx.Fasta, promoter: Promoter, input_length: int
) -> np.ndarray:
    half = input_length // 2
    tss0 = promoter.tss - 1
    start = tss0 - half
    end = start + input_length

    left_pad = max(0, -start)
    fetch_start = max(0, start)
    seq = str(fasta[promoter.chrom][fetch_start:end]).upper()
    seq = "N" * left_pad + seq
    seq = (seq + "N" * input_length)[:input_length]

    x = onehot_encode(seq)
    if promoter.strand in {"-", "-1"}:
        x = x[::-1, ::-1].copy()
    return x


def run_interleaved(args, x: np.ndarray, sample_names: list[str]) -> list:
    configure_tf_runtime(args.tf_device)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch_model, _ = load_pretrained(args.torch_checkpoint, map_location=str(device))
    torch_model.to(device).eval()
    tf_model = load_tf_model(args.keras_model)

    all_results = []
    for start in tqdm.trange(0, len(sample_names), args.batch_size, disable=not args.verbose):
        end = min(start + args.batch_size, len(sample_names))
        tf_outputs = predict_tf(tf_model, x[start:end])
        torch_outputs = predict_torch(torch_model, x[start:end], device)
        all_results.extend(
            compare_track_outputs(sample_names[start:end], tf_outputs, torch_outputs)
        )
        del tf_outputs, torch_outputs

    del tf_model, torch_model
    clear_tf_runtime()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return all_results


def write_tf_memmaps(args, x: np.ndarray, output_dir: Path) -> None:
    tf_model = load_tf_model(args.keras_model)
    tf_memmaps = None
    metadata = {"heads": []}
    for start in tqdm.trange(
        0,
        x.shape[0],
        args.batch_size,
        disable=not args.verbose,
        desc="TF",
    ):
        end = min(start + args.batch_size, x.shape[0])
        batch = np.asarray(x[start:end], dtype=np.float32)
        tf_outputs = predict_tf(tf_model, batch)
        if tf_memmaps is None:
            for head, out in enumerate(tf_outputs):
                path = output_dir / f"tf_head_{head}.npy"
                metadata["heads"].append(
                    {"path": path.name, "shape": [int(dim) for dim in out.shape[1:]]}
                )
            tf_memmaps = [
                np.lib.format.open_memmap(
                    output_dir / head["path"],
                    mode="w+",
                    dtype=np.float32,
                    shape=(x.shape[0], *head["shape"]),
                )
                for head in metadata["heads"]
            ]
        for memmap, out in zip(tf_memmaps, tf_outputs):
            memmap[start:end] = out
            memmap.flush()
        del tf_outputs

    with open(output_dir / "metadata.json", "w") as handle:
        json.dump(metadata, handle)
    del tf_model, tf_memmaps
    clear_tf_runtime()
    gc.collect()


def run_separate_loops(args, x: np.ndarray, sample_names: list[str]) -> list:
    with tempfile.TemporaryDirectory(prefix="promoterai_track_parity_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        x_path = tmp_path / "promoters.npy"
        np.save(x_path, x)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_tf_pass",
            "--_tf_input_npy",
            str(x_path),
            "--_tf_output_dir",
            str(tmp_path),
            "--keras_model",
            args.keras_model,
            "--torch_checkpoint",
            args.torch_checkpoint,
            "--fasta",
            args.fasta,
            "--input_length",
            str(args.input_length_resolved),
            "--batch_size",
            str(args.batch_size),
            "--tf_device",
            args.tf_device,
        ]
        if args.verbose:
            cmd.append("--verbose")
        subprocess.run(cmd, check=True)

        with open(tmp_path / "metadata.json") as handle:
            metadata = json.load(handle)
        tf_memmaps = [
            np.load(tmp_path / head["path"], mmap_mode="r")
            for head in metadata["heads"]
        ]

        device = torch.device(
            args.device
            if args.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch_model, _ = load_pretrained(args.torch_checkpoint, map_location=str(device))
        torch_model.to(device).eval()

        all_results = []
        for start in tqdm.trange(
            0,
            len(sample_names),
            args.batch_size,
            disable=not args.verbose,
            desc="Torch",
        ):
            end = min(start + args.batch_size, len(sample_names))
            tf_outputs = [np.asarray(memmap[start:end]) for memmap in tf_memmaps]
            torch_outputs = predict_torch(torch_model, x[start:end], device)
            all_results.extend(
                compare_track_outputs(sample_names[start:end], tf_outputs, torch_outputs)
            )
            del tf_outputs, torch_outputs

        del torch_model, tf_memmaps
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report mean/max absolute track error for TF vs Torch on promoters."
    )
    parser.add_argument("--keras_model", required=True, help="TF/Keras SavedModel path")
    parser.add_argument("--torch_checkpoint", required=True, help="Converted .pt path")
    parser.add_argument("--fasta", required=True, help="Reference FASTA path")
    parser.add_argument("--input_length", type=int, default=None)
    parser.add_argument(
        "--promoter",
        action="append",
        default=[],
        help="Promoter as NAME:CHROM:TSS:STRAND. May be repeated.",
    )
    parser.add_argument(
        "--promoter_tsv",
        action="append",
        default=[],
        help="TSV/TSV.GZ with chrom, tss_pos, strand, and optional gene. May be repeated.",
    )
    parser.add_argument(
        "--max_promoters_per_tsv",
        type=int,
        default=None,
        help="Optional cap after de-duplicating each promoter TSV.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--tf_device",
        choices=("cpu", "gpu"),
        default="gpu",
        help="TensorFlow device. GPU mode enables TensorFlow memory growth.",
    )
    parser.add_argument(
        "--separate_loops",
        dest="separate_loops",
        action="store_true",
        default=True,
        help=(
            "Run all TF predictions first in a child process, write them to "
            "temporary memmaps, then run Torch predictions."
        ),
    )
    parser.add_argument(
        "--interleaved",
        dest="separate_loops",
        action="store_false",
        help="Run TF and Torch predictions back-to-back for each batch.",
    )
    parser.add_argument("--_tf_pass", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_tf_input_npy", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_tf_output_dir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.input_length is not None:
        input_length = args.input_length
    else:
        checkpoint = torch.load(args.torch_checkpoint, map_location="cpu")
        model_args = checkpoint["args"]
        input_length = model_args.get("input_length")
        del checkpoint
        gc.collect()

    if input_length is None:
        raise ValueError(
            "--input_length is required when the Torch checkpoint lacks input_length metadata"
        )
    args.input_length_resolved = input_length

    if args._tf_pass:
        if args._tf_input_npy is None or args._tf_output_dir is None:
            raise ValueError(
                "--_tf_input_npy and --_tf_output_dir are required for the internal TF pass"
            )
        configure_tf_runtime(args.tf_device)
        x = np.load(args._tf_input_npy, mmap_mode="r")
        write_tf_memmaps(args, x, Path(args._tf_output_dir))
        return

    promoters = [parse_promoter(spec) for spec in args.promoter]
    for path in args.promoter_tsv:
        promoters.extend(promoters_from_tsv(path, limit=args.max_promoters_per_tsv))
    if not promoters:
        raise ValueError("Provide at least one --promoter or --promoter_tsv")

    fasta = pyfaidx.Fasta(args.fasta)
    x = np.stack(
        [sequence_for_promoter(fasta, promoter, input_length) for promoter in promoters]
    ).astype(np.float32)
    sample_names = [
        f"{p.name}:{p.chrom}:{p.tss}:{p.strand}" for p in promoters
    ]

    if args.separate_loops:
        all_results = run_separate_loops(args, x, sample_names)
    else:
        all_results = run_interleaved(args, x, sample_names)

    print_results(all_results)
    if args.output_csv:
        write_results_csv(Path(args.output_csv), all_results)


if __name__ == "__main__":
    main()
