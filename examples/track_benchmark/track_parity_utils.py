from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_REPORTED_MAPPING_ORDERS: set[tuple[str, ...]] = set()


@dataclass(frozen=True)
class TrackParityResult:
    sample: str
    head: int
    shape: tuple[int, ...]
    mean_abs_error: float
    max_abs_error: float


def load_tf_model(path: str):
    try:
        import tf_keras
    except ImportError as exc:
        raise ImportError(
            "tf-keras is required for TF/Torch track parity examples. "
            'Install with: pip install "promoterai-torch[convert]"'
        ) from exc

    return tf_keras.models.load_model(path)


def configure_tf_runtime(device: str = "cpu", memory_growth: bool = True) -> None:
    try:
        import tensorflow as tf
    except ImportError:
        return

    if device == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            print("Warning: TensorFlow runtime already initialized; could not hide GPUs.")
        return

    if memory_growth:
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                print(
                    "Warning: TensorFlow runtime already initialized; "
                    "could not enable memory growth."
                )
                break


def clear_tf_runtime() -> None:
    try:
        import tensorflow as tf
    except ImportError:
        return

    tf.keras.backend.clear_session()


def normalize_tf_outputs(outputs) -> list[np.ndarray]:
    if isinstance(outputs, Mapping):
        outputs = outputs_in_stable_order(outputs)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    return [np.asarray(out, dtype=np.float32) for out in outputs]


def normalize_torch_outputs(outputs) -> list[np.ndarray]:
    if isinstance(outputs, Mapping):
        outputs = outputs_in_stable_order(outputs)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    return [out.detach().cpu().numpy().astype(np.float32) for out in outputs]


def outputs_in_stable_order(outputs: Mapping) -> list:
    keys = list(outputs.keys())

    def _rank(key):
        key_str = str(key).lower()
        if "human" in key_str or "hg38" in key_str:
            return (0, key_str)
        if "mouse" in key_str or "mm10" in key_str:
            return (1, key_str)
        return (2, key_str)

    ordered_keys = sorted(keys, key=_rank)
    order = tuple(str(key) for key in ordered_keys)
    if order not in _REPORTED_MAPPING_ORDERS:
        print(f"Mapping model outputs in order: {', '.join(order)}")
        _REPORTED_MAPPING_ORDERS.add(order)
    return [outputs[key] for key in ordered_keys]


def predict_tf(tf_model, x: np.ndarray) -> list[np.ndarray]:
    return normalize_tf_outputs(tf_model(x, training=False))


def predict_torch(torch_model, x: np.ndarray, device: torch.device) -> list[np.ndarray]:
    x_t = torch.from_numpy(x).to(device)
    with torch.inference_mode():
        outputs = normalize_torch_outputs(torch_model(x_t))
    del x_t
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs


def compare_track_outputs(
    sample_names: list[str],
    tf_outputs: list[np.ndarray],
    torch_outputs: list[np.ndarray],
) -> list[TrackParityResult]:
    if len(tf_outputs) != len(torch_outputs):
        raise ValueError(
            f"TF produced {len(tf_outputs)} output heads but Torch produced "
            f"{len(torch_outputs)} output heads"
        )

    results: list[TrackParityResult] = []
    for head, (tf_out, torch_out) in enumerate(zip(tf_outputs, torch_outputs)):
        if tf_out.shape != torch_out.shape:
            raise ValueError(
                f"Head {head} shape mismatch: TF {tf_out.shape}, Torch {torch_out.shape}"
            )
        if tf_out.shape[0] != len(sample_names):
            raise ValueError(
                f"Head {head} batch size {tf_out.shape[0]} does not match "
                f"{len(sample_names)} sample names"
            )

        diff = np.abs(tf_out - torch_out)
        for i, sample in enumerate(sample_names):
            sample_diff = diff[i]
            results.append(
                TrackParityResult(
                    sample=sample,
                    head=head,
                    shape=tuple(tf_out[i].shape),
                    mean_abs_error=float(sample_diff.mean()),
                    max_abs_error=float(sample_diff.max()),
                )
            )
    return results


def print_results(results: Iterable[TrackParityResult]) -> None:
    rows = list(results)
    if not rows:
        print("No results.")
        return

    print("sample\thead\tshape\tmean_abs_error\tmax_abs_error")
    for row in rows:
        shape = "x".join(str(dim) for dim in row.shape)
        print(
            f"{row.sample}\t{row.head}\t{shape}\t"
            f"{row.mean_abs_error:.8g}\t{row.max_abs_error:.8g}"
        )

    mae = max(row.mean_abs_error for row in rows)
    max_err = max(row.max_abs_error for row in rows)
    print(f"\nWorst mean_abs_error: {mae:.8g}")
    print(f"Worst max_abs_error:  {max_err:.8g}")


def write_results_csv(path: str | Path, results: Iterable[TrackParityResult]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample", "head", "shape", "mean_abs_error", "max_abs_error"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "sample": row.sample,
                    "head": row.head,
                    "shape": "x".join(str(dim) for dim in row.shape),
                    "mean_abs_error": row.mean_abs_error,
                    "max_abs_error": row.max_abs_error,
                }
            )
