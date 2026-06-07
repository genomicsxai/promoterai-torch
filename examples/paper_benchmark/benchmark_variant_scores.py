#!/usr/bin/env python
from __future__ import annotations

import argparse
import math

from promoterai_torch.benchmark import run_benchmarks


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.4f}"
    return str(value)


def _split_devices(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    devices = []
    for value in values:
        devices.extend(part.strip() for part in value.split(",") if part.strip())
    return devices or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score PromoterAI benchmark TSVs with hg38_finetune and optional "
            "hg38_mm10_finetune torch checkpoints, then report AUROCs."
        )
    )
    parser.add_argument("--benchmark_dir", required=True)
    parser.add_argument("--hg38_finetune_checkpoint", required=True)
    parser.add_argument("--hg38_mm10_finetune_checkpoint", default=None)
    parser.add_argument("--fasta_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=(
            "Benchmark dataset to run, without or with .tsv extension. "
            "May be repeated. Default: run every TSV in --benchmark_dir."
        ),
    )
    parser.add_argument("--input_length", type=int, default=20480)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help=(
            "Devices for multi-process row sharding, e.g. --devices cuda:0 cuda:1 "
            "or --devices cuda:0,cuda:1. Overrides --device."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()

    rows = run_benchmarks(
        args.benchmark_dir,
        hg38_finetune_checkpoint=args.hg38_finetune_checkpoint,
        hg38_mm10_finetune_checkpoint=args.hg38_mm10_finetune_checkpoint,
        fasta_file=args.fasta_file,
        output_dir=args.output_dir,
        input_length=args.input_length,
        batch_size=args.batch_size,
        device=args.device,
        devices=_split_devices(args.devices),
        num_workers=args.num_workers,
        datasets=args.dataset,
        verbose=args.verbose,
    )

    columns = (
        "dataset",
        "n_under",
        "n_over",
        "n_null",
        "under_over_auroc",
        "under_null_auroc",
        "over_null_auroc",
    )
    print("\t".join(columns))
    for row in rows:
        print("\t".join(_fmt(row[col]) for col in columns))


if __name__ == "__main__":
    main()
