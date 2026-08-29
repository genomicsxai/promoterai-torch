from __future__ import annotations

import logging
import random as _random

import numpy as np
import pandas as pd
import pyfaidx
import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Sampler,
)

from promoterai_torch.onehot import onehot_encode


def _truncated_normal(stddev: float) -> float:
    """Sample from N(0, stddev) clipped to ±2*stddev via rejection sampling.

    Rejection-sampling a normal until it lands in [-2*stddev, 2*stddev] is the
    truncated normal distribution by definition, so this is equivalent to
    scipy.stats.truncnorm(-2, 2, scale=stddev).rvs() without the dependency —
    the 100-attempt cap is never practically reached (>95% acceptance rate per
    draw).
    """
    if stddev == 0:
        return 0.0
    for _ in range(100):
        v = _random.gauss(0, stddev)
        if abs(v) <= 2 * stddev:
            return v
    return 0.0


_logger = logging.getLogger(__name__)


def _prepare_sample(
    x: np.ndarray,
    y: np.ndarray,
    input_length: int,
    output_length: int,
    sample_weight: tuple,
    augment: bool,
) -> tuple:
    """Randomly re-center the over-length stored x/y around the model's window,
    optionally reverse-complementing both. Returns (x_crop, y_tuple, weight_tuple).
    """
    input_margin = x.shape[0] - input_length
    output_margin = y.shape[0] - output_length
    shift_span = min(input_margin, output_margin)
    max_shift = shift_span // 2

    shift = round(_truncated_normal(shift_span // 4 * int(augment)))
    shift = min(max(shift, -max_shift), max_shift)

    reverse_complement = bool(augment) and np.random.uniform() < 0.25
    strand = -1 if reverse_complement else 1

    x_start = input_margin // 2 + shift
    y_start = output_margin // 2 + shift
    x_crop = np.ascontiguousarray(
        x[x_start : x_start + input_length][::strand, ::strand]
    )
    y_crop = np.ascontiguousarray(y[y_start : y_start + output_length][::strand])

    y_tuple = tuple(
        y_crop if sw else np.array([[0.0]], dtype="float32") for sw in sample_weight
    )
    weight_tuple = tuple(float(sw) * float(x_crop.max()) for sw in sample_weight)
    return x_crop, y_tuple, weight_tuple


class SequenceDataset(Dataset):
    """Regulatory-track training dataset backed by preprocessed per-chromosome HDF5 files."""

    def __init__(
        self,
        hdf5_files: list,
        input_length: int,
        output_length: int,
        sample_weight: tuple,
        augment: bool = False,
    ):
        """Load sequences from HDF5 files; sample_weight is a per-species boolean tuple."""
        import h5py

        self.hdf5_files = hdf5_files
        self.input_length = input_length
        self.output_length = output_length
        self.sample_weight = sample_weight
        self.augment = augment
        # HDF5 handles are opened lazily per process/DataLoader worker. Opening
        # per sample is very expensive on shared filesystems.
        self._handles = {}
        # Build flat index: [(file_idx, row_idx), ...]
        self._index = []
        for fi, path in enumerate(hdf5_files):
            with h5py.File(path, "r") as f:  # h5py imported above
                n = f["x"].shape[0]
            self._index.extend((fi, ri) for ri in range(n))

    def __len__(self) -> int:
        """Return total number of samples across all HDF5 files."""
        return len(self._index)

    def __getstate__(self):
        """Drop open HDF5 handles before pickling so each DataLoader worker reopens its own."""
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def _get_handle(self, file_idx: int):
        """Return a cached read-only HDF5 handle for this worker process."""
        import h5py

        handle = self._handles.get(file_idx)
        if handle is None:
            handle = h5py.File(self.hdf5_files[file_idx], "r")
            self._handles[file_idx] = handle
        return handle

    def close(self) -> None:
        """Close any lazily opened HDF5 handles."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self):
        """Best-effort handle cleanup on garbage collection; ignores errors during interpreter teardown."""
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 - must never raise/log during interpreter teardown
            pass

    def __getitem__(self, idx: int):
        """Return (x_tensor, y_tuple, weight_tensor) after cropping and optional augmentation."""
        fi, ri = self._index[idx]
        handle = self._get_handle(fi)
        x = handle["x"][ri]  # (L_stored, 4)
        y = handle["y"][ri]  # (L_stored, n_tracks)
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


def _extract_window(
    fasta: pyfaidx.Fasta, chrom: str, pos: int, window: int, pad: bool
) -> str:
    """Return an uppercase window of `window` bases centered on `pos` (0-based).

    Bases past the chromosome edge are dropped by pyfaidx's slice; when `pad` is
    set, the missing bases are backfilled with 'N' (which one-hot-encodes to an
    all-zero row) so the result is always exactly `window` long. When `pad` is
    unset, a shorter-than-`window` string is returned as-is, signaling the
    caller that this locus fell off the chromosome edge.
    """
    half = window // 2
    lo, hi = pos - half, pos + half
    seq = str(fasta[chrom][max(lo, 0) : hi]).upper()
    if not pad or len(seq) >= window:
        return seq
    seq = "N" * max(0, -lo) + seq
    return (seq + "N" * window)[:window]


class VariantDataset(Dataset):
    """Dataset of ref/alt one-hot sequence pairs extracted from a FASTA around each variant."""

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

    def _fallback_label(self, row) -> float:
        """Return the label a skipped variant reports: 0.0 in 'zeros' mode, else the true label."""
        if self.boundary == "zeros" or not self.output_col:
            return 0.0
        return float(row[self.output_col])

    def __getitem__(self, idx: int):
        """Return ((x_ref, x_alt), y) where tensors are (input_length, 4); applies strand flip."""
        row = self.df.iloc[idx]
        chrom = row["chrom"]
        pos = int(row["pos"]) - 1  # 0-based
        ref_allele = row["ref"].upper()
        alt_allele = row["alt"].upper()
        strand = row.get("strand", 1)
        window = self.input_length
        center = window // 2

        def skip(reason: str):
            _logger.info(
                "variant %s:%s %s>%s skipped: %s",
                chrom,
                pos,
                ref_allele,
                alt_allele,
                reason,
            )
            placeholder = torch.zeros(window, 4)
            return (placeholder, placeholder), self._fallback_label(row)

        seq_ref = _extract_window(
            self.fasta, chrom, pos, window, pad=self.boundary == "pad"
        )
        if len(seq_ref) < window:
            return skip("too close to a chromosome edge")

        if seq_ref[center : center + len(ref_allele)] != ref_allele:
            return skip("reference allele does not match the genome")

        if not set(alt_allele) <= {"A", "C", "G", "T"}:
            return skip("alt allele contains non-ACGT characters")

        x_ref = onehot_encode(seq_ref)

        if len(alt_allele) == len(ref_allele):
            # Same-length substitution: edit only the changed positions instead of
            # re-extracting and re-encoding the whole window.
            x_alt = x_ref.copy()
            x_alt[center : center + len(alt_allele)] = onehot_encode(alt_allele)
        else:
            # Indel: downstream length shifts, so rebuild the full alt window.
            seq_alt = (
                seq_ref[:center] + alt_allele + seq_ref[center + len(ref_allele) :]
            )
            seq_alt = seq_alt.ljust(len(seq_ref), "N")[: len(seq_ref)]
            x_alt = onehot_encode(seq_alt)

        if strand in (-1, "-"):
            x_ref = np.ascontiguousarray(x_ref[::-1, ::-1])
            x_alt = np.ascontiguousarray(x_alt[::-1, ::-1])

        y = float(row[self.output_col]) if self.output_col else 0.0
        return (torch.from_numpy(x_ref), torch.from_numpy(x_alt)), y


def collate_variant(batch):
    """Stack ref/alt tensors and labels from a list of VariantDataset items into batched tensors."""
    x_refs = torch.stack([item[0][0] for item in batch])
    x_alts = torch.stack([item[0][1] for item in batch])
    ys = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    return (x_refs, x_alts), ys


class _SpeciesBatchSampler(Sampler):
    """Yield one batch of indices per step, drawn entirely from a single
    species dataset chosen with probability proportional to its size.

    Sampling rows independently across a ConcatDataset of species (the
    previous approach) lets a batch mix species with different real output
    shapes, which crashes the default collate_fn. Illumina's training loop
    instead runs tf.data.Dataset.sample_from_datasets over already-batched
    per-species streams, so every batch is single-species -- this mirrors
    that, at the same size-proportional sampling ratio.
    """

    def __init__(self, sizes: list, batch_size: int, num_batches: int, generator):
        self.sizes = sizes
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.generator = generator
        self.offsets = [0]
        for size in sizes[:-1]:
            self.offsets.append(self.offsets[-1] + size)

    def __len__(self) -> int:
        """Return the number of batches this sampler yields per epoch."""
        return self.num_batches

    def __iter__(self):
        """Yield num_batches lists of batch_size global indices, one species per batch."""
        weights = torch.tensor(self.sizes, dtype=torch.float64)
        species = torch.multinomial(
            weights, self.num_batches, replacement=True, generator=self.generator
        )
        for s in species.tolist():
            local = torch.randint(
                0, self.sizes[s], (self.batch_size,), generator=self.generator
            )
            yield (local + self.offsets[s]).tolist()


def build_weighted_dataloader(
    datasets: list,
    batch_size: int,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
    num_samples: int | None = None,
    prefetch_factor: int | None = 2,
) -> DataLoader:
    """
    Combine multiple SequenceDatasets. When shuffling, each batch is drawn
    entirely from one species dataset, chosen with probability proportional
    to that dataset's size (species with more data get sampled more often) --
    see _SpeciesBatchSampler.
    """
    sizes = [len(d) for d in datasets]
    total = sum(sizes)
    combined = ConcatDataset(datasets)

    worker_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        worker_kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            worker_kwargs["prefetch_factor"] = prefetch_factor

    if not shuffle:
        return DataLoader(
            combined,
            batch_size=batch_size,
            shuffle=False,
            **worker_kwargs,
        )

    samples_per_rank = num_samples if num_samples is not None else total
    num_batches = max(1, samples_per_rank // batch_size)
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed() + rank)
    batch_sampler = _SpeciesBatchSampler(sizes, batch_size, num_batches, generator)
    return DataLoader(
        combined,
        batch_sampler=batch_sampler,
        **worker_kwargs,
    )
