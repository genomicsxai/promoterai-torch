from __future__ import annotations

import numpy as np
import pandas as pd
import pyfaidx
import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)

try:
    from scipy.stats import truncnorm as _truncnorm

    def _truncated_normal(stddev: float) -> float:
        """Sample from N(0, stddev) clipped to ±2*stddev using scipy for accuracy."""
        if stddev == 0:
            return 0.0
        return float(_truncnorm.rvs(-2, 2, scale=stddev))
except ImportError:
    import random as _random

    def _truncated_normal(stddev: float) -> float:
        """Sample from N(0, stddev) clipped to ±2*stddev via rejection sampling."""
        if stddev == 0:
            return 0.0
        for _ in range(100):
            v = _random.gauss(0, stddev)
            if abs(v) <= 2 * stddev:
                return v
        return 0.0


_EMBED = np.zeros((26, 4), dtype="float32")
_EMBED[ord("A") - 65] = [1, 0, 0, 0]
_EMBED[ord("C") - 65] = [0, 1, 0, 0]
_EMBED[ord("G") - 65] = [0, 0, 1, 0]
_EMBED[ord("T") - 65] = [0, 0, 0, 1]


def onehot_encode(seq: str) -> np.ndarray:
    """One-hot encode a DNA string. Unknown bases → all-zero row. Returns (L, 4) float32."""
    seq = seq.upper()
    idx = np.frombuffer(seq.encode(), dtype=np.uint8).astype(np.int32) - 65
    idx = np.clip(idx, 0, 25)
    return _EMBED[idx]


def _prepare_sample(
    x: np.ndarray,
    y: np.ndarray,
    input_length: int,
    output_length: int,
    sample_weight: tuple,
    augment: bool,
) -> tuple:
    """Port of tfrecords.py::_prepare_sample. Returns (x_crop, y_tuple, weight_tuple)."""
    input_crop = x.shape[0] - input_length
    output_crop = y.shape[0] - output_length
    shift_range = min(input_crop, output_crop)
    shift = int(round(_truncated_normal(shift_range // 4 * int(augment))))
    shift = max(-shift_range // 2, min(shift_range // 2, shift))

    strand = -1 if (augment and np.random.uniform() < 0.25) else 1

    ic2 = input_crop // 2
    oc2 = output_crop // 2
    # When shift == ic2 the end index would be 0, which is wrong — use None instead
    x_end = shift - ic2 if shift != ic2 else None
    y_end = shift - oc2 if shift != oc2 else None
    x_crop = x[shift + ic2 : x_end][::strand, ::strand]
    y_crop = y[shift + oc2 : y_end][::strand]

    y_tuple = tuple(
        y_crop if sw else np.array([[0.0]], dtype="float32") for sw in sample_weight
    )
    weight_tuple = tuple(float(sw) * float(x_crop.max()) for sw in sample_weight)
    return x_crop, y_tuple, weight_tuple


class SequenceDataset(Dataset):
    def __init__(
        self,
        hdf5_files: list,
        input_length: int,
        output_length: int,
        sample_weight: tuple,
        augment: bool = False,
    ):
        """Load sequences from HDF5 files; sample_weight is a per-species boolean tuple."""
        import h5py  # noqa: PLC0415 — optional train dependency

        self.hdf5_files = hdf5_files
        self.input_length = input_length
        self.output_length = output_length
        self.sample_weight = sample_weight
        self.augment = augment
        # Build flat index: [(file_idx, row_idx), ...]
        self._index = []
        for fi, path in enumerate(hdf5_files):
            with h5py.File(path, "r") as f:  # h5py imported above
                n = f["x"].shape[0]
            self._index.extend((fi, ri) for ri in range(n))

    def __len__(self) -> int:
        """Return total number of samples across all HDF5 files."""
        return len(self._index)

    def __getitem__(self, idx: int):
        """Return (x_tensor, y_tuple, weight_tensor) after cropping and optional augmentation."""
        import h5py

        fi, ri = self._index[idx]
        with h5py.File(self.hdf5_files[fi], "r") as f:
            x = f["x"][ri]  # (L_stored, 4)
            y = f["y"][ri]  # (L_stored, n_tracks)
        x_crop, y_tuple, w_tuple = _prepare_sample(
            x,
            y,
            self.input_length,
            self.output_length,
            self.sample_weight,
            self.augment,
        )
        x_t = torch.from_numpy(x_crop)
        y_t = tuple(torch.from_numpy(np.array(yt, dtype="float32")) for yt in y_tuple)
        w_t = torch.tensor(w_tuple, dtype=torch.float32)
        return x_t, y_t, w_t


class VariantDataset(Dataset):
    def __init__(
        self,
        df_var: pd.DataFrame,
        fasta: pyfaidx.Fasta,
        input_length: int,
        output_col: str | None = None,
        shuffle: bool = False,
        boundary: str = "pad",
    ):
        """Dataset of ref/alt sequence pairs; mismatched ref alleles fall back to zero tensors.

        boundary: how to handle variants within input_length//2 bases of a chromosome edge.
          'pad'   — N-pad the extracted sequence to input_length (N encodes as all-zero row).
          'zeros' — return all-zero tensors, matching the reference TF implementation.
        """
        if boundary not in ("pad", "zeros"):
            raise ValueError(f"boundary must be 'pad' or 'zeros', got {boundary!r}")
        self.df = df_var.reset_index(drop=True)
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        # Store the path rather than the open Fasta object so that each DataLoader
        # worker process opens its own file handle after fork, preventing fd races.
        self._fasta_path = (
            fasta.filename if isinstance(fasta, pyfaidx.Fasta) else str(fasta)
        )
        self._fasta: pyfaidx.Fasta | None = None
        self.input_length = input_length
        self.output_col = output_col
        self.boundary = boundary

    @property
    def fasta(self) -> pyfaidx.Fasta:
        """Open-on-first-access Fasta handle, once per process."""
        if self._fasta is None:
            self._fasta = pyfaidx.Fasta(self._fasta_path)
        return self._fasta

    def __len__(self) -> int:
        """Return number of variants."""
        return len(self.df)

    def __getitem__(self, idx: int):
        """Return ((x_ref, x_alt), y) where tensors are (input_length, 4); applies strand flip."""
        row = self.df.iloc[idx]
        chrom, pos = row["chrom"], int(row["pos"]) - 1  # 0-based
        ref_allele = row["ref"].upper()
        alt_allele = row["alt"].upper()
        strand = row.get("strand", 1)
        half = self.input_length // 2

        start = pos - half
        seq_ref_str = str(self.fasta[chrom][max(0, start) : pos + half]).upper()
        center = half

        def _zero():
            x = torch.zeros(self.input_length, 4)
            y = float(row[self.output_col]) if self.output_col else 0.0
            return (x, x), y

        if len(seq_ref_str) < self.input_length:
            if self.boundary == "zeros":
                print(f"Skipping {chrom}:{pos} {ref_allele}>{alt_allele} (pos issue)")
                return _zero()
            # pad: left-pad with N then right-pad; N encodes as all-zero row
            left_pad = max(0, -start)
            seq_ref_str = "N" * left_pad + seq_ref_str
            seq_ref_str = (seq_ref_str + "N" * self.input_length)[: self.input_length]

        extracted_ref = seq_ref_str[center : center + len(ref_allele)]
        if extracted_ref != ref_allele:
            print(f"Skipping {chrom}:{pos} {ref_allele}>{alt_allele} (ref issue)")
            return _zero()

        if not set(alt_allele).issubset({"A", "C", "G", "T"}):
            print(f"Skipping {chrom}:{pos} {ref_allele}>{alt_allele} (alt issue)")
            return _zero()

        # Construct alt sequence
        seq_alt_str = (
            seq_ref_str[:center] + alt_allele + seq_ref_str[center + len(ref_allele) :]
        )
        # Pad or truncate alt to same length as ref
        seq_alt_str = seq_alt_str.ljust(len(seq_ref_str), "N")[: len(seq_ref_str)]

        x_ref = onehot_encode(seq_ref_str)
        x_alt = onehot_encode(seq_alt_str)

        # Strand flip
        if strand in (-1, "-"):
            x_ref = x_ref[::-1, ::-1].copy()
            x_alt = x_alt[::-1, ::-1].copy()

        y = float(row[self.output_col]) if self.output_col else 0.0
        return (torch.from_numpy(x_ref), torch.from_numpy(x_alt)), y


def build_weighted_dataloader(
    datasets: list,
    batch_size: int,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
    num_samples: int | None = None,
) -> DataLoader:
    """
    Combine multiple SequenceDatasets with weights proportional to dataset size,
    matching tf.data.Dataset.sample_from_datasets(datasets, weights=output_size).
    """
    sizes = [len(d) for d in datasets]
    total = sum(sizes)
    combined = ConcatDataset(datasets)

    if not shuffle:
        return DataLoader(
            combined,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    # Equal per-sample weights produce dataset-level probabilities size_i / total,
    # matching tf.data.Dataset.sample_from_datasets(..., weights=dataset_sizes).
    weights = torch.ones(total, dtype=torch.float64)
    samples_per_rank = num_samples if num_samples is not None else total
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed() + rank)
    sampler = WeightedRandomSampler(
        weights, num_samples=samples_per_rank, replacement=True, generator=generator
    )
    return DataLoader(
        combined,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
