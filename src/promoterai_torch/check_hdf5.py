"""Integrity checks for PromoterAI-torch HDF5 training chunks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

HDF5_SUFFIXES = {".h5", ".hdf5", ".hdf"}


@dataclass(frozen=True)
class HDF5CheckResult:
    path: Path
    ok: bool
    samples: int = 0
    error: str | None = None


def find_hdf5_files(paths: Iterable[str | Path]) -> list[Path]:
    """Return sorted HDF5 files found under the provided files/folders."""
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.suffix.lower() in HDF5_SUFFIXES:
            files.append(path)
        elif path.is_dir():
            files.extend(
                child
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in HDF5_SUFFIXES
            )
    return sorted(files)


def check_hdf5_file(path: str | Path, full_read: bool = False) -> HDF5CheckResult:
    """Validate one HDF5 file and force basic dataset reads."""
    path = Path(path)
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            if "x" not in handle:
                raise ValueError("missing required dataset 'x'")
            if "y" not in handle:
                raise ValueError("missing required dataset 'y'")

            x = handle["x"]
            y = handle["y"]
            _validate_shapes(x.shape, y.shape)

            if full_read:
                x[:]
                y[:]
            elif x.shape[0] > 0:
                x[0]
                x[-1]
                y[0]
                y[-1]

            return HDF5CheckResult(path=path, ok=True, samples=int(x.shape[0]))
    except Exception as exc:  # noqa: BLE001 - report h5py/value errors uniformly.
        return HDF5CheckResult(path=path, ok=False, error=f"{type(exc).__name__}: {exc}")


def check_hdf5_paths(
    paths: Iterable[str | Path], full_read: bool = False
) -> tuple[list[HDF5CheckResult], int]:
    """Check all HDF5 files under paths and return results plus process exit code."""
    files = find_hdf5_files(paths)
    if not files:
        return [], 2

    results = [check_hdf5_file(path, full_read=full_read) for path in files]
    exit_code = 1 if any(not result.ok for result in results) else 0
    return results, exit_code


def print_summary(results: list[HDF5CheckResult]) -> None:
    """Print a concise human-readable integrity summary."""
    bad = [result for result in results if not result.ok]
    total_samples = sum(result.samples for result in results if result.ok)
    print(f"total files checked: {len(results)}")
    print(f"total samples: {total_samples}")
    print(f"bad file count: {len(bad)}")
    for result in bad:
        print(f"BAD {result.path}: {result.error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check PromoterAI HDF5 training chunks for corruption."
    )
    parser.add_argument("--paths", nargs="+", required=True, help="HDF5 files/folders")
    parser.add_argument(
        "--full-read",
        "--full_read",
        action="store_true",
        help="Read every x/y value instead of only first and last rows",
    )
    args = parser.parse_args(argv)

    results, exit_code = check_hdf5_paths(args.paths, full_read=args.full_read)
    if exit_code == 2:
        print("no HDF5 files found")
        return exit_code

    print_summary(results)
    return exit_code


def _validate_shapes(x_shape: tuple[int, ...], y_shape: tuple[int, ...]) -> None:
    if len(x_shape) != 3:
        raise ValueError(f"x must have shape (n, input_len, 4); got {x_shape}")
    if len(y_shape) != 3:
        raise ValueError(f"y must have shape (n, output_len, tracks); got {y_shape}")
    if x_shape[2] != 4:
        raise ValueError(f"x last dimension must be 4; got {x_shape}")
    if x_shape[1] <= 0:
        raise ValueError(f"x input length must be positive; got {x_shape}")
    if y_shape[1] <= 0 or y_shape[2] <= 0:
        raise ValueError(f"y output length and tracks must be positive; got {y_shape}")
    if x_shape[0] != y_shape[0]:
        raise ValueError(
            f"x/y sample counts differ: x has {x_shape[0]}, y has {y_shape[0]}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
