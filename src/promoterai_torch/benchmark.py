from __future__ import annotations

import csv
import math
import multiprocessing as mp
import queue as queue_module
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from promoterai_torch.architecture import TwinModel
from promoterai_torch.dataset import VariantDataset, collate_variant
from promoterai_torch.utils import DEFAULT_INPUT_LENGTH, load_pretrained

BENCHMARK_BASE_URL = (
    "https://raw.githubusercontent.com/Illumina/PromoterAI/master/data/benchmark"
)
BENCHMARK_FILES = (
    "CAGI5_saturation.tsv",
    "GEL_RNA.tsv",
    "GTEx_eQTL.tsv",
    "GTEx_outlier.tsv",
    "MPRA_eQTL.tsv",
    "MPRA_saturation.tsv",
    "UKBB_proteome.tsv",
)
SCORE_COLUMNS = ("hg38_finetune_score", "hg38_mm10_finetune_score")


def _score_jobs(
    hg38_finetune_checkpoint: str | Path,
    hg38_mm10_finetune_checkpoint: str | Path | None,
) -> tuple[tuple[str, str | Path], ...]:
    """Return (output_column, checkpoint) pairs for the checkpoints that were provided."""
    jobs: list[tuple[str, str | Path]] = [
        ("hg38_finetune_score", hg38_finetune_checkpoint)
    ]
    if hg38_mm10_finetune_checkpoint is not None:
        jobs.append(("hg38_mm10_finetune_score", hg38_mm10_finetune_checkpoint))
    return tuple(jobs)


def download_benchmark_data(
    output_dir: str | Path,
    *,
    files: Iterable[str] = BENCHMARK_FILES,
    base_url: str = BENCHMARK_BASE_URL,
    overwrite: bool = False,
) -> list[tuple[Path, bool]]:
    """Download PromoterAI benchmark TSVs and return (path, downloaded) pairs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for name in files:
        path = output_dir / name
        if path.exists() and not overwrite:
            paths.append((path, False))
            continue
        urllib.request.urlretrieve(f"{base_url}/{name}", path)
        paths.append((path, True))
    return paths


def score_variants(
    df_var: pd.DataFrame,
    *,
    model_checkpoint: str | Path,
    fasta_file: str | Path,
    input_length: int,
    batch_size: int = 2,
    device: str | None = None,
    num_workers: int = 4,
    verbose: bool = False,
) -> np.ndarray:
    """Score a variant dataframe with one torch checkpoint and return tanh-scaled scores."""
    if len(df_var) == 0:
        return np.array([], dtype=np.float32)

    torch_device = torch.device(
        device if device else "cuda" if torch.cuda.is_available() else "cpu"
    )
    base_model, _ = load_pretrained(str(model_checkpoint), map_location=str(torch_device))
    twin_model = TwinModel(base_model).to(torch_device)
    twin_model.eval()

    dataset = VariantDataset(df_var, str(fasta_file), input_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_variant,
    )

    all_diffs = []
    with torch.no_grad():
        batches = loader
        if verbose:
            from tqdm import tqdm

            batches = tqdm(loader, desc=Path(model_checkpoint).stem, unit="batch")
        for (x_ref, x_alt), _ in batches:
            diff = twin_model(x_ref.to(torch_device), x_alt.to(torch_device))
            all_diffs.append(diff.cpu().numpy())

    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return np.tanh(np.concatenate(all_diffs).round(4))


def benchmark_paths(
    benchmark_dir: str | Path,
    datasets: Iterable[str] | None = None,
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


def score_benchmark_file(
    benchmark_tsv: str | Path,
    *,
    hg38_finetune_checkpoint: str | Path,
    hg38_mm10_finetune_checkpoint: str | Path | None = None,
    fasta_file: str | Path,
    input_length: int,
    batch_size: int = 2,
    device: str | None = None,
    num_workers: int = 4,
    verbose: bool = False,
) -> pd.DataFrame:
    """Score one benchmark TSV and add the single-model or ensemble score."""
    df = pd.read_csv(benchmark_tsv, sep="\t")
    score_columns = []
    for col, checkpoint in _score_jobs(
        hg38_finetune_checkpoint, hg38_mm10_finetune_checkpoint
    ):
        score_columns.append(col)
        df[col] = score_variants(
            df,
            model_checkpoint=checkpoint,
            fasta_file=fasta_file,
            input_length=input_length,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
            verbose=verbose,
        )
    df["score"] = df[score_columns].mean(axis=1)
    return df


def _score_shard_worker(
    queue,
    worker_idx: int,
    shard: pd.DataFrame,
    device: str,
    hg38_finetune_checkpoint: str,
    hg38_mm10_finetune_checkpoint: str | None,
    fasta_file: str,
    input_length: int,
    batch_size: int,
    num_workers: int,
    verbose: bool,
) -> None:
    """Score one dataframe shard in a child process and report arrays through a queue."""
    try:
        result = {"worker_idx": worker_idx, "index": shard.index.to_numpy()}
        for col, checkpoint in _score_jobs(
            hg38_finetune_checkpoint, hg38_mm10_finetune_checkpoint
        ):
            result[col] = score_variants(
                shard.reset_index(drop=True),
                model_checkpoint=checkpoint,
                fasta_file=fasta_file,
                input_length=input_length,
                batch_size=batch_size,
                device=device,
                num_workers=num_workers,
                verbose=verbose,
            )
        queue.put(("ok", result))
    except BaseException as exc:  # noqa: BLE001 - report every worker failure, including SystemExit, through the queue
        queue.put(("error", worker_idx, repr(exc)))


def score_benchmark_file_multi_device(
    benchmark_tsv: str | Path,
    *,
    hg38_finetune_checkpoint: str | Path,
    hg38_mm10_finetune_checkpoint: str | Path | None = None,
    fasta_file: str | Path,
    input_length: int,
    devices: list[str],
    batch_size: int = 2,
    num_workers: int = 0,
    verbose: bool = False,
) -> pd.DataFrame:
    """Score one benchmark TSV by row-sharding it over multiple devices."""
    if not devices:
        raise ValueError("devices must contain at least one device")
    if len(devices) == 1:
        return score_benchmark_file(
            benchmark_tsv,
            hg38_finetune_checkpoint=hg38_finetune_checkpoint,
            hg38_mm10_finetune_checkpoint=hg38_mm10_finetune_checkpoint,
            fasta_file=fasta_file,
            input_length=input_length,
            batch_size=batch_size,
            device=devices[0],
            num_workers=num_workers,
            verbose=verbose,
        )

    df = pd.read_csv(benchmark_tsv, sep="\t")
    shard_indices = np.array_split(np.arange(len(df)), len(devices))
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []

    active_shards = [
        (device, indices) for device, indices in zip(devices, shard_indices) if len(indices)
    ]
    for worker_idx, (device, indices) in enumerate(active_shards):
        shard = df.iloc[indices].copy()
        process = ctx.Process(
            target=_score_shard_worker,
            args=(
                queue,
                worker_idx,
                shard,
                device,
                str(hg38_finetune_checkpoint),
                (
                    str(hg38_mm10_finetune_checkpoint)
                    if hg38_mm10_finetune_checkpoint is not None
                    else None
                ),
                str(fasta_file),
                input_length,
                batch_size,
                num_workers,
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
                        "One or more benchmark workers exited before reporting: "
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
        raise RuntimeError(f"One or more benchmark workers exited non-zero: {failed}")

    score_columns = [
        col
        for col, _ in _score_jobs(
            hg38_finetune_checkpoint, hg38_mm10_finetune_checkpoint
        )
    ]
    for col in score_columns:
        df[col] = np.nan
    for result in results:
        for col in score_columns:
            df.loc[result["index"], col] = result[col]
    df["score"] = df[score_columns].mean(axis=1)
    return df


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


def run_benchmarks(
    benchmark_dir: str | Path,
    *,
    hg38_finetune_checkpoint: str | Path,
    hg38_mm10_finetune_checkpoint: str | Path | None = None,
    fasta_file: str | Path,
    output_dir: str | Path,
    input_length: int = DEFAULT_INPUT_LENGTH,
    batch_size: int = 2,
    device: str | None = None,
    devices: list[str] | None = None,
    num_workers: int = 4,
    datasets: Iterable[str] | None = None,
    verbose: bool = False,
) -> list[dict[str, object]]:
    """Score every downloaded benchmark TSV and write scored TSVs plus metrics."""
    benchmark_dir = Path(benchmark_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    resolved_devices = devices if devices is not None else ([device] if device else None)
    for benchmark_tsv in benchmark_paths(benchmark_dir, datasets):
        score_kwargs = {
            "hg38_finetune_checkpoint": hg38_finetune_checkpoint,
            "hg38_mm10_finetune_checkpoint": hg38_mm10_finetune_checkpoint,
            "fasta_file": fasta_file,
            "input_length": input_length,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "verbose": verbose,
        }
        if resolved_devices is not None:
            scored = score_benchmark_file_multi_device(
                benchmark_tsv,
                devices=resolved_devices,
                **score_kwargs,
            )
        else:
            scored = score_benchmark_file(
                benchmark_tsv,
                device=device,
                **score_kwargs,
            )
        out_tsv = output_dir / f"{benchmark_tsv.stem}.scores.tsv"
        scored.to_csv(out_tsv, sep="\t", index=False)

        row = {"dataset": benchmark_tsv.stem}
        row.update(benchmark_aurocs(scored))
        rows.append(row)

    write_metrics(rows, output_dir / "benchmark_aurocs.tsv")
    return rows
