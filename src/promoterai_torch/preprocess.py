"""
Preprocess TSS annotations + genome FASTA + BigWig tracks into one HDF5 file per
chromosome for the PyTorch training pipeline. The BigWig TSV must contain `fwd`,
`rev`, and `xform` columns: paths to the forward- and reverse-strand BigWig files
for a track, and a Python expression (e.g. "np.arcsinh") applied to its values.

Samples are written incrementally in --chunk_size batches into a single growable
HDF5 file per chromosome (rather than one small file per batch), bounding peak
memory use without producing a large number of small files per chromosome.

Usage:
    python -m promoterai_torch.preprocess \
        --hdf5_folder <out> --tss_file data/annotation/tss_hg38.tsv \
        --fasta_file <genome.fa> --bigwig_files data/bigwig/hg38.tsv \
        --chrom chr1 --input_length 32768 --output_length 16384 --chunk_size 256
"""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyfaidx

from promoterai_torch.onehot import onehot_encode

if TYPE_CHECKING:
    import h5py

_pybigtools_import_error = None
try:
    import pybigtools
except ImportError as exc:
    pybigtools = None
    _pybigtools_import_error = exc


def _extract_bigwig(bw, chrom, start, end):
    """Fetch values from a BigWig handle; returns zeros on any error or missing region."""
    try:
        vals = bw.values(chrom, max(0, start), end, missing=0.0, oob=0.0)
        if vals is None:
            return np.zeros(end - start, dtype="float32")
        vals = np.array(vals, dtype="float32")
        return vals
    except Exception:  # noqa: BLE001 - any pybigtools read failure should fall back to zeros
        return np.zeros(end - start, dtype="float32")


def _compile_xform(expr: str):
    """Compile a per-track BigWig value transform expression (e.g. "np.arcsinh") from the bigwig TSV."""
    return eval(expr, {"np": np, "__builtins__": {}})


def _create_growable_dataset(
    f: h5py.File, name: str, sample_shape: tuple, chunk_rows: int
) -> h5py.Dataset:
    """Create a resizable, gzip-compressed dataset that starts empty and grows by row."""
    return f.create_dataset(
        name,
        shape=(0, *sample_shape),
        maxshape=(None, *sample_shape),
        dtype="float32",
        chunks=(chunk_rows, *sample_shape),
        compression="gzip",
        compression_opts=4,
    )


def _append_rows(dataset: h5py.Dataset, rows: np.ndarray) -> None:
    """Grow dataset by len(rows) and write them into the newly added slice."""
    start = dataset.shape[0]
    dataset.resize(start + len(rows), axis=0)
    dataset[start:] = rows


def preprocess_chrom(
    tss_file: str,
    fasta_file: str,
    bigwig_files_tsv: str,
    chrom: str,
    hdf5_folder: str,
    input_length: int,
    output_length: int,
    chunk_size: int,
):
    """Extract sequences and BigWig tracks for all TSS on chrom and write one HDF5 file."""
    import h5py  # optional train dependency, kept out of module-level imports

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
    if pybigtools is None:
        raise ImportError(
            "pybigtools is required for preprocessing. "
            'Install with: pip install "promoterai-torch[train]"'
        ) from _pybigtools_import_error
    bws_fwd = [pybigtools.open(p) for p in df_bw["fwd"]]
    bws_rev = [pybigtools.open(p) for p in df_bw["rev"]]
    bw_xforms = [_compile_xform(expr) for expr in df_bw["xform"]]

    half_in = input_length // 2
    half_out = output_length // 2
    n_tracks = len(bws_fwd)

    os.makedirs(hdf5_folder, exist_ok=True)
    hdf5_path = os.path.join(hdf5_folder, f"{chrom}.h5")
    h5_file: h5py.File | None = None
    x_ds = y_ds = None
    xs_buf, ys_buf = [], []
    total_rows = 0

    def flush(force=False):
        """Append buffered samples to the chromosome's HDF5 file, creating it on first flush."""
        nonlocal xs_buf, ys_buf, h5_file, x_ds, y_ds, total_rows
        if not xs_buf or (not force and len(xs_buf) < chunk_size):
            return
        xs = np.stack(xs_buf).astype("float32")
        ys = np.stack(ys_buf).astype("float32")
        if h5_file is None:
            h5_file = h5py.File(hdf5_path, "w")
            x_ds = _create_growable_dataset(h5_file, "x", xs.shape[1:], chunk_size)
            y_ds = _create_growable_dataset(h5_file, "y", ys.shape[1:], chunk_size)
        _append_rows(x_ds, xs)
        _append_rows(y_ds, ys)
        total_rows += len(xs_buf)
        print(f"Wrote {len(xs_buf)} samples ({total_rows} total) to {hdf5_path}")
        xs_buf, ys_buf = [], []

    for _, row in df_tss.iterrows():
        pos = int(row["pos"]) - 1  # 0-based
        strand = row.get("strand", 1)

        seq_str = str(fasta[chrom][pos - half_in : pos + half_in]).upper()
        if len(seq_str) < input_length:
            # Too close to a chromosome edge for a full window. Illumina's
            # generator.py keeps this TSS as an all-zero row (rather than
            # dropping it) so it stays in the dataset and gets zero-weighted
            # downstream via _prepare_sample's x.max()==0 check; mirror that
            # instead of shrinking the dataset and skewing per-species sizes.
            xs_buf.append(np.zeros((input_length, 4), dtype="float32"))
            ys_buf.append(np.zeros((output_length, n_tracks), dtype="float32"))
            if len(xs_buf) >= chunk_size:
                flush()
            continue
        x = onehot_encode(seq_str)

        is_minus = strand in (-1, "-1", "-")
        tracks = []
        for j, (bw_fwd, bw_rev, xform) in enumerate(zip(bws_fwd, bws_rev, bw_xforms)):
            bw = bw_rev if is_minus else bw_fwd
            vals = _extract_bigwig(bw, chrom, pos - half_out, pos + half_out)
            if len(vals) < output_length:
                vals = np.pad(vals, (0, output_length - len(vals)))
            vals = xform(vals[:output_length])
            vals = np.asarray(vals, dtype="float32")
            tracks.append(vals)
        y = np.stack(tracks, axis=-1)  # (output_length, n_tracks)

        if is_minus:
            x = x[::-1, ::-1].copy()
            y = y[::-1, :].copy()

        xs_buf.append(x)
        ys_buf.append(y)

        if len(xs_buf) >= chunk_size:
            flush()

    flush(force=True)
    if h5_file is not None:
        h5_file.close()

    for bw in bws_fwd + bws_rev:
        bw.close()

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
        default=256,
        help="Number of samples buffered in memory before each incremental write "
        "to the chromosome's HDF5 file (default: %(default)s)",
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
    )


if __name__ == "__main__":
    main()
