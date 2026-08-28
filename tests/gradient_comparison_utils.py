"""Cosine-similarity / relative-L2 tensor comparison utilities.

Adapted from alphagenome-pytorch's "gradient ladder" methodology for comparing
JAX vs. PyTorch gradients: comparing whole tensors by direction (cosine
similarity) and magnitude (relative L2 norm), then requiring a pass *rate*
across many tensors rather than every element of every tensor, is far more
robust to float32 noise on near-zero-gradient elements (e.g. a value that
lands on the wrong side of a ReLU boundary due to summation-order differences
between frameworks) than elementwise np.testing.assert_allclose, which blows
up on tiny denominators and fails outright on a single such element.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ComparisonResult:
    """Direction (cosine similarity) and magnitude (relative L2) agreement between
    two same-shaped tensors, treated as flattened vectors.
    """

    name: str
    cosine_sim: float
    rel_l2: float
    max_abs_diff: float
    pt_norm: float
    ref_norm: float

    def passed(self, cosine_threshold: float = 0.99, rel_l2_tol: float = 0.05) -> bool:
        """Whether this tensor agrees within the given cosine/relative-L2 thresholds."""
        return self.cosine_sim >= cosine_threshold and self.rel_l2 <= rel_l2_tol


def compare_tensors(name: str, pt: np.ndarray, ref: np.ndarray) -> ComparisonResult:
    """Compare two same-shaped arrays by cosine similarity and relative L2 norm.

    `ref` is the reference (e.g. the Keras/JAX side); `rel_l2` is normalized by
    `ref`'s norm so it reads as "PyTorch differs from the reference by X%".
    """
    pt_flat = np.asarray(pt).ravel().astype(np.float64)
    ref_flat = np.asarray(ref).ravel().astype(np.float64)
    if pt_flat.shape != ref_flat.shape:
        raise ValueError(f"{name}: shape mismatch {pt_flat.shape} vs {ref_flat.shape}")

    pt_norm = float(np.linalg.norm(pt_flat))
    ref_norm = float(np.linalg.norm(ref_flat))
    if pt_norm > 0 and ref_norm > 0:
        cosine_sim = float(np.dot(pt_flat, ref_flat) / (pt_norm * ref_norm))
    else:
        cosine_sim = 1.0 if pt_norm == ref_norm == 0 else 0.0
    rel_l2 = float(np.linalg.norm(pt_flat - ref_flat) / (ref_norm + 1e-12))
    max_abs_diff = float(np.max(np.abs(pt_flat - ref_flat)))
    return ComparisonResult(name, cosine_sim, rel_l2, max_abs_diff, pt_norm, ref_norm)


def compare_all(
    pt_tensors: dict, ref_tensors: dict, names: list | None = None
) -> list[ComparisonResult]:
    """Compare every named tensor present in both dicts (or just `names` if given)."""
    keys = names if names is not None else [k for k in ref_tensors if k in pt_tensors]
    return [compare_tensors(name, pt_tensors[name], ref_tensors[name]) for name in keys]


def report_top_offenders(results: list[ComparisonResult], k: int = 10) -> str:
    """Format the k worst-agreeing tensors (lowest cosine similarity) for debugging."""
    worst = sorted(results, key=lambda r: r.cosine_sim)[:k]
    lines = [f"Top {len(worst)} worst by cosine similarity (of {len(results)} total):"]
    for r in worst:
        lines.append(
            f"  {r.name}: cosine={r.cosine_sim:.4f} rel_l2={r.rel_l2:.4%} "
            f"max_diff={r.max_abs_diff:.3g} (pt_norm={r.pt_norm:.3g}, ref_norm={r.ref_norm:.3g})"
        )
    return "\n".join(lines)


def assert_pass_rate(
    results: list[ComparisonResult],
    *,
    cosine_threshold: float = 0.99,
    rel_l2_tol: float = 0.05,
    min_pass_rate: float = 1.0,
    label: str = "",
) -> None:
    """Assert at least `min_pass_rate` of `results` pass the cosine/rel-L2 thresholds.

    Prints a top-offenders report on failure so a real regression is easy to localize
    even though this check tolerates a minority of noisy outliers.
    """
    if not results:
        return
    passed = [r for r in results if r.passed(cosine_threshold, rel_l2_tol)]
    rate = len(passed) / len(results)
    assert rate >= min_pass_rate, (
        f"{label}: only {rate:.1%} of {len(results)} tensors passed "
        f"(cosine>={cosine_threshold}, rel_l2<={rel_l2_tol:.1%}); "
        f"expected >= {min_pass_rate:.1%}\n" + report_top_offenders(results)
    )
