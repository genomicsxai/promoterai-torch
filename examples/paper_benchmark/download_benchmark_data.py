#!/usr/bin/env python
from __future__ import annotations

import argparse

from promoterai_torch.benchmark import BENCHMARK_FILES, download_benchmark_data


def _dataset_files(values: list[str] | None) -> tuple[str, ...]:
    if values is None:
        return BENCHMARK_FILES
    files = []
    expected = set(BENCHMARK_FILES)
    for value in values:
        name = value if value.endswith(".tsv") else f"{value}.tsv"
        if name not in expected:
            raise SystemExit(f"Unknown benchmark dataset: {value}")
        files.append(name)
    return tuple(files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the public Illumina PromoterAI benchmark TSVs."
    )
    parser.add_argument(
        "--output_dir",
        default="../data/paper_benchmarks/",
        help="Directory where benchmark TSVs will be written.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=(
            "Benchmark dataset to download, without or with .tsv extension. "
            "May be repeated. Default: download every benchmark TSV."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files that already exist.",
    )
    args = parser.parse_args()

    results = download_benchmark_data(
        args.output_dir,
        files=_dataset_files(args.dataset),
        overwrite=args.overwrite,
    )
    expected = set(BENCHMARK_FILES)
    for path, downloaded in results:
        status = "downloaded" if downloaded else "exists"
        if path.name not in expected:
            status = "written"
        print(f"{status}: {path}")


if __name__ == "__main__":
    main()
