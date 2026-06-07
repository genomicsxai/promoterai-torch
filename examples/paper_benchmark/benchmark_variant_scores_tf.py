#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import queue as queue_module
from pathlib import Path

import numpy as np
import pandas as pd

SCORE_COLUMNS = ("hg38_finetune_score", "hg38_mm10_finetune_score")


def _score_jobs(
    hg38_finetune_model_folder: str | Path,
    hg38_mm10_finetune_model_folder: str | Path | None,
) -> tuple[tuple[str, str | Path], ...]:
    jobs: list[tuple[str, str | Path]] = [
        ("hg38_finetune_score", hg38_finetune_model_folder)
    ]
    if hg38_mm10_finetune_model_folder is not None:
        jobs.append(("hg38_mm10_finetune_score", hg38_mm10_finetune_model_folder))
    return tuple(jobs)


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


def benchmark_paths(
    benchmark_dir: str | Path,
    datasets: list[str] | None = None,
) -> list[Path]:
    """Resolve benchmark TSV paths, optionally restricted to named datasets."""
    benchmark_dir = Path(benchmark_dir)
    if datasets is None:
        return sorted(benchmark_dir.glob("*.tsv"))

    paths = []
    for dataset in datasets:
        path = Path(dataset)
        if not path.is_absolute():
            path = benchmark_dir / path
        if path.suffix != ".tsv":
            path = path.with_suffix(".tsv")
        if not path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found: {path}")
        paths.append(path)
    return paths


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC with average ranks for ties; returns NaN for one-class input."""
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    valid = ~np.isnan(scores)
    y_true = y_true[valid]
    scores = scores[valid]

    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    rank_sum_pos = ranks[y_true].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def benchmark_aurocs(
    df: pd.DataFrame,
    *,
    score_col: str = "score",
    label_col: str = "consequence",
) -> dict[str, float]:
    """Return under/over, under/null, and over/null AUROCs for one scored benchmark."""
    labels = df[label_col].astype(str).str.lower()
    scores = df[score_col].to_numpy(dtype=float)

    under_over = labels.isin(("under", "over"))
    under_null = labels.isin(("under", "none", "null"))
    over_null = labels.isin(("over", "none", "null"))
    null_labels = labels.isin(("none", "null"))

    return {
        "under_over_auroc": auroc(labels[under_over].eq("over"), scores[under_over]),
        "under_null_auroc": auroc(
            labels[under_null].eq("under"), -scores[under_null]
        ),
        "over_null_auroc": auroc(labels[over_null].eq("over"), scores[over_null]),
        "n_under": int(labels.eq("under").sum()),
        "n_over": int(labels.eq("over").sum()),
        "n_null": int(null_labels.sum()),
    }


def write_metrics(metrics: list[dict[str, object]], output_tsv: str | Path) -> None:
    """Write benchmark metric rows to a TSV."""
    output_tsv = Path(output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "n_under",
        "n_over",
        "n_null",
        "under_over_auroc",
        "under_null_auroc",
        "over_null_auroc",
    ]
    with output_tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(metrics)


def _configure_visible_device(device: str | None) -> str | None:
    """Set CUDA_VISIBLE_DEVICES before TensorFlow import; return a TF device name."""
    if device is None:
        return None
    normalized = device.strip().lower()
    if normalized in {"cpu", "cpu:0", "/cpu:0"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return "/CPU:0"
    if normalized.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = normalized
        return "/GPU:0"
    for prefix in ("cuda:", "gpu:", "/gpu:"):
        if normalized.startswith(prefix):
            os.environ["CUDA_VISIBLE_DEVICES"] = normalized.rsplit(":", 1)[1]
            return "/GPU:0"
    return device


def _load_tf_runtime(device: str | None):
    tf_device = _configure_visible_device(device)
    import pyfaidx
    import tensorflow as tf
    import tensorflow.keras as tk
    from promoterai.architecture import twin_wrap
    from promoterai.generator import VariantDataGenerator

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return pyfaidx, tf, tk, twin_wrap, VariantDataGenerator, tf_device


def _pad_for_generator(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    if len(df) == 0 or batch_size <= 1:
        return df
    remainder = len(df) % batch_size
    if remainder == 0:
        return df
    pad_n = batch_size - remainder
    padding = pd.concat([df.tail(1)] * pad_n, ignore_index=True)
    return pd.concat([df, padding], ignore_index=True)


def score_variants_tf(
    df_var: pd.DataFrame,
    *,
    model_folder: str | Path,
    fasta_file: str | Path,
    input_length: int,
    batch_size: int = 1,
    device: str | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Score variants through the official TensorFlow PromoterAI package."""
    if len(df_var) == 0:
        return np.array([], dtype=np.float32)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    pyfaidx, tf, tk, twin_wrap, VariantDataGenerator, tf_device = _load_tf_runtime(
        device
    )
    fasta = pyfaidx.Fasta(str(fasta_file))
    padded = _pad_for_generator(df_var.reset_index(drop=True), batch_size)
    generator = VariantDataGenerator(padded, fasta, input_length, batch_size)

    with tf.device(tf_device) if tf_device else _nullcontext():
        model = tk.models.load_model(str(model_folder))
        twin_model = twin_wrap(model)
        predictions = twin_model.predict(generator, verbose=1 if verbose else 0)

    scores = np.tanh(np.asarray(predictions).reshape(-1)[: len(df_var)].round(4))
    return scores


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc_info):
        return False


def score_benchmark_file_tf(
    benchmark_tsv: str | Path,
    *,
    hg38_finetune_model_folder: str | Path,
    hg38_mm10_finetune_model_folder: str | Path | None = None,
    fasta_file: str | Path,
    input_length: int,
    batch_size: int = 1,
    device: str | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(benchmark_tsv, sep="\t")
    score_columns = []
    for col, model_folder in _score_jobs(
        hg38_finetune_model_folder, hg38_mm10_finetune_model_folder
    ):
        score_columns.append(col)
        df[col] = score_variants_tf(
            df,
            model_folder=model_folder,
            fasta_file=fasta_file,
            input_length=input_length,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )
    df["score"] = df[score_columns].mean(axis=1)
    return df


def _score_tf_shard_worker(
    queue,
    worker_idx: int,
    shard: pd.DataFrame,
    device: str,
    hg38_finetune_model_folder: str,
    hg38_mm10_finetune_model_folder: str | None,
    fasta_file: str,
    input_length: int,
    batch_size: int,
    verbose: bool,
) -> None:
    try:
        result = {"worker_idx": worker_idx, "index": shard.index.to_numpy()}
        for col, model_folder in _score_jobs(
            hg38_finetune_model_folder, hg38_mm10_finetune_model_folder
        ):
            result[col] = score_variants_tf(
                shard.reset_index(drop=True),
                model_folder=model_folder,
                fasta_file=fasta_file,
                input_length=input_length,
                batch_size=batch_size,
                device=device,
                verbose=verbose,
            )
        queue.put(("ok", result))
    except BaseException as exc:
        queue.put(("error", worker_idx, repr(exc)))


def score_benchmark_file_tf_multi_device(
    benchmark_tsv: str | Path,
    *,
    hg38_finetune_model_folder: str | Path,
    hg38_mm10_finetune_model_folder: str | Path | None = None,
    fasta_file: str | Path,
    input_length: int,
    devices: list[str],
    batch_size: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    if not devices:
        raise ValueError("devices must contain at least one device")
    if len(devices) == 1:
        return score_benchmark_file_tf(
            benchmark_tsv,
            hg38_finetune_model_folder=hg38_finetune_model_folder,
            hg38_mm10_finetune_model_folder=hg38_mm10_finetune_model_folder,
            fasta_file=fasta_file,
            input_length=input_length,
            batch_size=batch_size,
            device=devices[0],
            verbose=verbose,
        )

    df = pd.read_csv(benchmark_tsv, sep="\t")
    shard_indices = np.array_split(np.arange(len(df)), len(devices))
    active_shards = [
        (device, indices) for device, indices in zip(devices, shard_indices) if len(indices)
    ]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []

    for worker_idx, (device, indices) in enumerate(active_shards):
        process = ctx.Process(
            target=_score_tf_shard_worker,
            args=(
                queue,
                worker_idx,
                df.iloc[indices].copy(),
                device,
                str(hg38_finetune_model_folder),
                (
                    str(hg38_mm10_finetune_model_folder)
                    if hg38_mm10_finetune_model_folder is not None
                    else None
                ),
                str(fasta_file),
                input_length,
                batch_size,
                verbose,
            ),
        )
        process.start()
        processes.append(process)

    results = []
    should_terminate = False
    try:
        while len(results) < len(processes):
            try:
                message = queue.get(timeout=1)
            except queue_module.Empty:
                dead = [p for p in processes if p.exitcode not in (None, 0)]
                if dead:
                    should_terminate = True
                    raise RuntimeError(
                        "One or more TF benchmark workers exited before reporting: "
                        f"{[p.exitcode for p in dead]}"
                    )
                continue
            if message[0] == "error":
                should_terminate = True
                raise RuntimeError(f"Worker {message[1]} failed: {message[2]}")
            results.append(message[1])
    finally:
        if should_terminate:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join()

    failed = [process.exitcode for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"One or more TF benchmark workers exited non-zero: {failed}")

    score_columns = [
        col
        for col, _ in _score_jobs(
            hg38_finetune_model_folder, hg38_mm10_finetune_model_folder
        )
    ]
    for col in score_columns:
        df[col] = np.nan
    for result in results:
        for col in score_columns:
            df.loc[result["index"], col] = result[col]
    df["score"] = df[score_columns].mean(axis=1)
    return df


def run_tf_benchmarks(
    benchmark_dir: str | Path,
    *,
    hg38_finetune_model_folder: str | Path,
    hg38_mm10_finetune_model_folder: str | Path | None = None,
    fasta_file: str | Path,
    output_dir: str | Path,
    input_length: int = 20480,
    batch_size: int = 1,
    device: str | None = None,
    devices: list[str] | None = None,
    datasets: list[str] | None = None,
    verbose: bool = False,
) -> list[dict[str, object]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_devices = devices if devices is not None else ([device] if device else None)

    rows = []
    for benchmark_tsv in benchmark_paths(benchmark_dir, datasets):
        score_kwargs = dict(
            hg38_finetune_model_folder=hg38_finetune_model_folder,
            hg38_mm10_finetune_model_folder=hg38_mm10_finetune_model_folder,
            fasta_file=fasta_file,
            input_length=input_length,
            batch_size=batch_size,
            verbose=verbose,
        )
        if resolved_devices is not None:
            scored = score_benchmark_file_tf_multi_device(
                benchmark_tsv,
                devices=resolved_devices,
                **score_kwargs,
            )
        else:
            scored = score_benchmark_file_tf(
                benchmark_tsv,
                device=device,
                **score_kwargs,
            )
        out_tsv = output_dir / f"{benchmark_tsv.stem}.tensorflow.scores.tsv"
        scored.to_csv(out_tsv, sep="\t", index=False)

        row = {"dataset": benchmark_tsv.stem}
        row.update(benchmark_aurocs(scored))
        rows.append(row)

    write_metrics(rows, output_dir / "benchmark_aurocs.tensorflow.tsv")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark official TensorFlow PromoterAI SavedModels on the public "
            "benchmark TSVs and report AUROCs."
        )
    )
    parser.add_argument("--benchmark_dir", required=True)
    parser.add_argument("--hg38_finetune_model_folder", required=True)
    parser.add_argument("--hg38_mm10_finetune_model_folder", default=None)
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Official VariantDataGenerator batch size. Defaults to 1.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help=(
            "Devices for multi-process row sharding, e.g. --devices 0 1 or "
            "--devices cuda:0,cuda:1. Overrides --device."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()

    rows = run_tf_benchmarks(
        args.benchmark_dir,
        hg38_finetune_model_folder=args.hg38_finetune_model_folder,
        hg38_mm10_finetune_model_folder=args.hg38_mm10_finetune_model_folder,
        fasta_file=args.fasta_file,
        output_dir=args.output_dir,
        input_length=args.input_length,
        batch_size=args.batch_size,
        device=args.device,
        devices=_split_devices(args.devices),
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
