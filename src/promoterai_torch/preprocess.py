"""
Preprocess TSS annotations + genome FASTA + BigWig tracks into one HDF5 file per
chromosome for the PyTorch training pipeline. The BigWig TSV must contain `fwd`,
`rev`, and `xform` columns: paths to the forward- and reverse-strand BigWig files
for a track, and a Python expression (e.g. "np.arcsinh") applied to its values.

Samples are buffered in --chunk_size batches (each batch's BigWig windows are
read in one multithreaded figwig call) and appended to a single growable HDF5
file per chromosome, stored one sample per HDF5 chunk for fast random-row reads
during training. The y buffer is chunk_size x output_length x n_tracks float32
(64 x 16384 x 498 is ~2 GiB), so lower --chunk_size for many tracks.

Both expensive steps are multithreaded with --threads (default: the CPUs this
process may use): figwig decompresses BigWig blocks on that many threads, and
each sample's HDF5 chunk is gzip-compressed on a thread pool (zlib releases the
GIL) and stored with write_direct_chunk -- byte-for-byte what h5py's own gzip
filter writes, so readers are unaffected. Chromosomes are independent, so
several can also run at once (e.g. with GNU parallel); keep processes x
--threads at about the core count. Threads share one buffer while every process
holds its own, so prefer fewer processes with more threads when memory is tight.

Usage:
    python -m promoterai_torch.preprocess \
        --hdf5_folder <out> --tss_file data/annotation/tss_hg38.tsv \
        --fasta_file <genome.fa> --bigwig_files data/bigwig/hg38.tsv \
        --chrom chr1 --input_length 32768 --output_length 16384 --chunk_size 64
"""

from __future__ import annotations

import argparse
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyfaidx

from promoterai_torch.onehot import onehot_encode

if TYPE_CHECKING:
    import h5py

_figwig_import_error = None
try:
    import figwig
except ImportError as exc:
    figwig = None
    _figwig_import_error = exc

_GZIP_LEVEL = 4

# Target uncompressed bytes per HDF5 chunk. Each sample is split along its length
# into chunks of about this size, so a training read of a cropped window only
# decompresses the chunks it overlaps.
_CHUNK_BYTES = 1 << 20


def _chunk_length(length: int, n_features: int) -> int:
    """Positions per HDF5 chunk: a power of two near _CHUNK_BYTES of float32, <= length."""
    positions = max(1, _CHUNK_BYTES // (4 * n_features))
    return min(length, 1 << (positions.bit_length() - 1))


def _available_cpus() -> int:
    """CPUs this process may actually use, not the node's total core count.

    Slurm's per-task allocation if set, else the CPU affinity mask (which
    reflects taskset and Slurm/cgroup cpusets); os.cpu_count() would report
    every core on the node and oversubscribe a restricted job.
    """
    slurm = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if slurm.isdigit() and int(slurm) > 0:
        return int(slurm)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on macOS/Windows
        return os.cpu_count() or 1


def _read_tracks(readers, chrom, starts, width, n_jobs=1):
    """Read one window per start from every reader, as (n, len(readers), width).

    Uncovered bases and a chrom absent from a file are 0. Bases past the
    chromosome end are NaN in figwig's output and are left that way here;
    `_finish_sample` turns them into 0, matching pybigtools'
    `values(..., missing=0.0, oob=0.0)`. Every start must be >= 0.
    """
    starts = np.asarray(starts, dtype=np.int64)
    out = np.empty((len(starts), len(readers), width), dtype="float32")
    if len(starts):
        figwig.read_bigwig(list(readers), chrom, starts, width, out=out, n_jobs=n_jobs)
    return out


def _finish_sample(tracks: np.ndarray, xforms, minus: bool, out: np.ndarray) -> None:
    """Turn one sample's raw (n_tracks, width) BigWig values into its y row.

    Zero-fills past-chromosome-end NaNs, applies each track's xform, then
    flips (minus strand) and transposes into out (width, n_tracks). Runs on the
    thread pool, one sample per task: numpy releases the GIL for these
    elementwise ops and copies, and samples write disjoint rows.
    """
    np.copyto(tracks, 0.0, where=np.isnan(tracks))
    for j, xform in enumerate(xforms):
        tracks[j] = xform(tracks[j])
    out[...] = (tracks[:, ::-1] if minus else tracks).T


def _compile_xform(expr: str):
    """Compile a per-track BigWig value transform expression (e.g. "np.arcsinh") from the bigwig TSV.

    The transform is applied to a whole (n_windows, output_length) block of one
    track at a time, so it must be elementwise (any numpy ufunc or arithmetic
    lambda is).
    """
    return eval(expr, {"np": np, "__builtins__": {}})


def _create_growable_dataset(f: h5py.File, name: str, sample_shape: tuple) -> h5py.Dataset:
    """Create a resizable, gzip-compressed dataset that starts empty and grows by row.

    Each HDF5 chunk is part of one sample: one row, and about _CHUNK_BYTES along
    the length. HDF5 decompresses a whole chunk to read any part of it, and
    training reads one random row at a time and keeps only a window of it (y is
    stored ~4x longer than the model's output, for shift augmentation). Multi-row
    chunks made every read decompress many other samples (67x slower per read at
    64 rows per chunk) and can exceed HDF5's 4 GB chunk limit at real scale;
    whole-row chunks made every read decompress the full stored length (~2.7x
    slower for a 4096-bp window of a 16384-bp row than ~1-2 kb chunks).
    """
    length, n_features = sample_shape
    return f.create_dataset(
        name,
        shape=(0, *sample_shape),
        maxshape=(None, *sample_shape),
        dtype="float32",
        chunks=(1, _chunk_length(length, n_features), n_features),
        compression="gzip",
        compression_opts=_GZIP_LEVEL,
    )


def _append_rows(
    dataset: h5py.Dataset, rows: np.ndarray, pool: ThreadPoolExecutor | None = None
) -> None:
    """Grow dataset by len(rows) and write them into the newly added rows.

    With a pool, each chunk (part of one row) is deflated on the pool's threads
    with zlib at the dataset's gzip level and stored as-is via
    write_direct_chunk. That gives the same chunk bytes h5py's gzip filter
    would write itself, single-threaded. A row's last chunk, when the length is
    not a multiple of the chunk length, is zero-padded to the full chunk size,
    as HDF5 stores edge chunks (0 is the dataset's fill value).
    """
    start = dataset.shape[0]
    dataset.resize(start + len(rows), axis=0)
    if pool is None:
        dataset[start:] = rows
        return
    rows = np.ascontiguousarray(rows, dtype=dataset.dtype)
    _, chunk_len, n_features = dataset.chunks
    length = rows.shape[1]

    def piece(i, k):
        part = rows[i, k : k + chunk_len]
        if len(part) < chunk_len:
            part = np.concatenate([part, np.zeros((chunk_len - len(part), n_features), part.dtype)])
        return zlib.compress(memoryview(np.ascontiguousarray(part)), _GZIP_LEVEL)

    offsets = [(i, k) for i in range(len(rows)) for k in range(0, length, chunk_len)]
    for (i, k), blob in zip(offsets, pool.map(lambda ik: piece(*ik), offsets)):
        dataset.id.write_direct_chunk((start + i, k, 0), blob)


def preprocess_chrom(
    tss_file: str,
    fasta_file: str,
    bigwig_files_tsv: str,
    chrom: str,
    hdf5_folder: str,
    input_length: int,
    output_length: int,
    chunk_size: int,
    threads: int | None = None,
):
    """Extract sequences and BigWig tracks for all TSS on chrom and write one HDF5 file.

    threads: threads for BigWig decompression and HDF5 gzip compression
    (default: the CPUs available to this process; 1 disables the thread pool).
    """
    import h5py  # optional train dependency, kept out of module-level imports

    threads = _available_cpus() if threads is None else max(1, int(threads))

    df_tss = pd.read_csv(tss_file, sep="\t")
    df_tss = df_tss[df_tss["chrom"] == chrom].reset_index(drop=True)
    if df_tss.empty:
        print(f"No TSS entries for {chrom}, skipping.")
        return

    df_bw = pd.read_csv(bigwig_files_tsv, sep="\t")
    required_cols = {"fwd", "rev", "xform"}
    missing = required_cols - set(df_bw.columns)
    if missing:
        raise ValueError(
            f"bigwig_files must contain columns {sorted(required_cols)}, "
            f"missing {sorted(missing)}"
        )

    fasta = pyfaidx.Fasta(fasta_file)
    if figwig is None:
        raise ImportError(
            "figwig is required for preprocessing. "
            'Install with: pip install "promoterai-torch[train]"'
        ) from _figwig_import_error
    bws_fwd = [figwig.BigWigReader(p) for p in df_bw["fwd"]]
    bws_rev = [figwig.BigWigReader(p) for p in df_bw["rev"]]
    bw_xforms = [_compile_xform(expr) for expr in df_bw["xform"]]

    half_in = input_length // 2
    half_out = output_length // 2
    n_tracks = len(bws_fwd)

    os.makedirs(hdf5_folder, exist_ok=True)
    hdf5_path = os.path.join(hdf5_folder, f"{chrom}.h5")
    h5_file: h5py.File | None = None
    x_ds = y_ds = None
    # One entry per buffered sample: its one-hot x (None for a boundary
    # zero row), BigWig window start, and strand.
    buf: list[tuple[np.ndarray | None, int, bool]] = []
    total_rows = 0

    def build_y() -> np.ndarray:
        """Batch-read the buffered samples' tracks; (n, output_length, n_tracks).

        Each strand's windows are read in one multithreaded figwig call into a
        (k, n_tracks, output_length) scratch array; each sample is then
        finished (NaN fill, xform, flip, transpose) into its row of the output
        on the thread pool. Peak memory is the output plus one strand's
        scratch. Boundary rows stay all-zero, even when xform(0) != 0.
        """
        ys = np.zeros((len(buf), output_length, n_tracks), dtype="float32")
        run = pool.map if pool is not None else map
        for minus, readers in ((False, bws_fwd), (True, bws_rev)):
            idx = [i for i, (x, _, m) in enumerate(buf) if x is not None and m == minus]
            if not idx:
                continue
            tracks = _read_tracks(
                readers, chrom, [buf[i][1] for i in idx], output_length, n_jobs=threads
            )
            # list() drains the iterator so every task finishes (and raises) here.
            list(run(lambda k: _finish_sample(tracks[k], bw_xforms, minus, ys[idx[k]]),
                     range(len(idx))))
        return ys

    def flush(force=False):
        """Append buffered samples to the chromosome's HDF5 file, creating it on first flush."""
        nonlocal buf, h5_file, x_ds, y_ds, total_rows
        if not buf or (not force and len(buf) < chunk_size):
            return
        ys = build_y()
        xs = np.zeros((len(buf), input_length, 4), dtype="float32")
        for i, (x, _, _) in enumerate(buf):
            if x is not None:
                xs[i] = x
        if h5_file is None:
            h5_file = h5py.File(hdf5_path, "w")
            x_ds = _create_growable_dataset(h5_file, "x", xs.shape[1:])
            y_ds = _create_growable_dataset(h5_file, "y", ys.shape[1:])
        _append_rows(x_ds, xs, pool)
        _append_rows(y_ds, ys, pool)
        total_rows += len(buf)
        print(f"Wrote {len(buf)} samples ({total_rows} total) to {hdf5_path}")
        buf = []

    pool = ThreadPoolExecutor(threads) if threads > 1 else None
    try:
        strands = df_tss["strand"] if "strand" in df_tss.columns else [1] * len(df_tss)
        for pos, strand in zip(df_tss["pos"], strands):
            pos = int(pos) - 1  # 0-based
            is_minus = strand in (-1, "-1", "-")
            bw_start = pos - half_out

            seq_str = (
                str(fasta[chrom][pos - half_in : pos + half_in]).upper()
                if pos - half_in >= 0
                else ""
            )
            if len(seq_str) < input_length or bw_start < 0:
                # Too close to a chromosome edge for a full window. Illumina's
                # generator.py keeps this TSS as an all-zero row (rather than
                # dropping it) so it stays in the dataset and gets zero-weighted
                # downstream via _prepare_sample's x.max()==0 check; mirror that
                # instead of shrinking the dataset and skewing per-species sizes.
                buf.append((None, 0, is_minus))
            else:
                x = onehot_encode(seq_str)
                buf.append((x[::-1, ::-1] if is_minus else x, bw_start, is_minus))

            if len(buf) >= chunk_size:
                flush()

        flush(force=True)
    finally:
        if pool is not None:
            pool.shutdown()
        if h5_file is not None:
            h5_file.close()

    if total_rows:
        print(f"Preprocessing complete for {chrom}: {total_rows} samples written to {hdf5_path}.")
    else:
        print(f"Preprocessing complete for {chrom}: no samples passed filtering, no file written.")


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Build (or populate, when composed into the unified CLI) the preprocess parser."""
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf5_folder", required=True, help="Output folder for per-chromosome HDF5 files"
    )
    parser.add_argument(
        "--tss_file",
        required=True,
        help="TSV of TSS positions to center training windows on",
    )
    parser.add_argument(
        "--fasta_file", required=True, help="Reference genome FASTA (indexed with pyfaidx)"
    )
    parser.add_argument(
        "--bigwig_files",
        required=True,
        help="TSV mapping track names to BigWig file paths",
    )
    parser.add_argument("--chrom", required=True, help="Chromosome to preprocess (e.g. chr1)")
    parser.add_argument(
        "--input_length", type=int, required=True, help="Input sequence length in bp"
    )
    parser.add_argument(
        "--output_length", type=int, required=True, help="Output track length in bp"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="Number of samples buffered in memory, BigWig-read in one batch, and "
        "appended to the chromosome's HDF5 file together; the y buffer is "
        "chunk_size x output_length x n_tracks float32 (default: %(default)s)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Threads for BigWig decompression and HDF5 gzip compression "
        "(default: the CPUs available to this process -- SLURM_CPUS_PER_TASK, "
        "else the CPU affinity mask). When running several chromosomes at once, "
        "keep processes x threads at about the core count.",
    )
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    """Parse args (if not already parsed) and run preprocess_chrom for the specified chromosome."""
    if args is None:
        args = build_parser().parse_args()

    os.makedirs(args.hdf5_folder, exist_ok=True)
    preprocess_chrom(
        tss_file=args.tss_file,
        fasta_file=args.fasta_file,
        bigwig_files_tsv=args.bigwig_files,
        chrom=args.chrom,
        hdf5_folder=args.hdf5_folder,
        input_length=args.input_length,
        output_length=args.output_length,
        chunk_size=args.chunk_size,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
