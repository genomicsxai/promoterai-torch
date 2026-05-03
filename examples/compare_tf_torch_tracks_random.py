#!/usr/bin/env python
"""
Compare TF/Keras and PyTorch PromoterAI predicted tracks on random DNA sequences.

Example:
    python examples/compare_tf_torch_tracks_random.py \
        --keras_model models/promoterAI_v1_hg38_mm10_finetune \
        --torch_checkpoint models/promoterAI_v1_hg38_mm10_finetune.pt \
        --n_sequences 8
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import tqdm

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


def random_onehot(n_sequences: int, input_length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bases = rng.integers(0, 4, size=(n_sequences, input_length))
    return np.eye(4, dtype=np.float32)[bases]


def run_interleaved(args, x: np.ndarray, sample_names: list[str]) -> list:
    configure_tf_runtime(args.tf_device)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch_model, _ = load_pretrained(args.torch_checkpoint, map_location=str(device))
    torch_model.to(device).eval()
    tf_model = load_tf_model(args.keras_model)

    all_results = []
    for start in tqdm.trange(0, args.n_sequences, args.batch_size, disable=not args.verbose):
        end = min(start + args.batch_size, args.n_sequences)
        batch = x[start:end]
        batch_names = sample_names[start:end]
        tf_outputs = predict_tf(tf_model, batch)
        torch_outputs = predict_torch(torch_model, batch, device)
        all_results.extend(compare_track_outputs(batch_names, tf_outputs, torch_outputs))
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
        args.n_sequences,
        args.batch_size,
        disable=not args.verbose,
        desc="TF",
    ):
        end = min(start + args.batch_size, args.n_sequences)
        tf_outputs = predict_tf(tf_model, x[start:end])
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
                    shape=(args.n_sequences, *head["shape"]),
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


def run_separate_loops(
    args, x: np.ndarray, sample_names: list[str], input_length: int
) -> list:
    with tempfile.TemporaryDirectory(prefix="promoterai_track_parity_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_tf_pass",
            "--_tf_output_dir",
            str(tmp_path),
            "--keras_model",
            args.keras_model,
            "--torch_checkpoint",
            args.torch_checkpoint,
            "--input_length",
            str(input_length),
            "--n_sequences",
            str(args.n_sequences),
            "--batch_size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
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
            args.n_sequences,
            args.batch_size,
            disable=not args.verbose,
            desc="Torch",
        ):
            end = min(start + args.batch_size, args.n_sequences)
            batch_names = sample_names[start:end]
            tf_outputs = [np.asarray(memmap[start:end]) for memmap in tf_memmaps]
            torch_outputs = predict_torch(torch_model, x[start:end], device)
            all_results.extend(compare_track_outputs(batch_names, tf_outputs, torch_outputs))
            del tf_outputs, torch_outputs

        del torch_model, tf_memmaps
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report mean/max absolute track error for TF vs Torch on random sequences.",
    )
    parser.add_argument("--keras_model", required=True, help="TF/Keras SavedModel path")
    parser.add_argument("--torch_checkpoint", required=True, help="Converted .pt path")
    parser.add_argument("--input_length", type=int, default=None)
    parser.add_argument("--n_sequences", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
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
            "Run all TF predictions first, write them to temporary memmaps, clear TF, "
            "then run Torch predictions."
        ),
    )
    parser.add_argument(
        "--interleaved",
        dest="separate_loops",
        action="store_false",
        help="Run TF and Torch predictions back-to-back for each batch.",
    )
    parser.add_argument("--_tf_pass", action="store_true", help=argparse.SUPPRESS)
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

    x = random_onehot(args.n_sequences, input_length, args.seed)
    sample_names = [f"random_{i}" for i in range(args.n_sequences)]

    if args._tf_pass:
        if args._tf_output_dir is None:
            raise ValueError("--_tf_output_dir is required for the internal TF pass")
        configure_tf_runtime(args.tf_device)
        write_tf_memmaps(args, x, Path(args._tf_output_dir))
        return

    if args.separate_loops:
        all_results = run_separate_loops(args, x, sample_names, input_length)
    else:
        all_results = run_interleaved(args, x, sample_names)

    print_results(all_results)
    if args.output_csv:
        write_results_csv(Path(args.output_csv), all_results)


if __name__ == "__main__":
    main()
