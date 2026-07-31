"""
Preprocess TSS annotations + genome FASTA + BigWig tracks into HDF5 files.

This mirrors the original PromoterAI TFRecord preprocessing semantics, but writes
HDF5 files for the PyTorch training pipeline. The BigWig TSV must contain
`fwd`, `rev`, and `xform` columns.

Usage:
    python -m promoterai_torch.preprocess \
        --hdf5_folder <out> --tss_file data/annotation/tss_hg38.tsv \
        --fasta_file <genome.fa> --bigwig_files data/bigwig/hg38.tsv \
        --chrom chr1 --input_length 32768 --output_length 16384 --chunk_size 256
"""

import argparse
import os

import h5py
import numpy as np
import pandas as pd
import pyfaidx

from promoterai_torch.dataset import onehot_encode

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
    except Exception:
        return np.zeros(end - start, dtype="float32")


def _compile_xform(expr: str):
    """Compile a BigWig transform expression from the original PromoterAI TSV."""
    return eval(expr, {"np": np, "__builtins__": {}})  # noqa: S307 - original API is expressions


def make_hdf5_file(hdf5_file: str, xs: np.ndarray, ys: np.ndarray):
    """Write x and y arrays to an HDF5 file with gzip compression level 4."""
    os.makedirs(os.path.dirname(hdf5_file) or ".", exist_ok=True)
    with h5py.File(hdf5_file, "w") as f:
        f.create_dataset(
            "x", data=xs.astype("float32"), compression="gzip", compression_opts=4
        )
        f.create_dataset(
            "y", data=ys.astype("float32"), compression="gzip", compression_opts=4
        )


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
    """Extract sequences and BigWig tracks for all TSS on chrom and write chunked HDF5 files."""
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

    xs_buf, ys_buf = [], []
    chunk_idx = 0

    def flush(force=False):
        """Write buffered samples to disk when chunk_size is reached or force=True."""
        nonlocal xs_buf, ys_buf, chunk_idx
        if not xs_buf:
            return
        if force or len(xs_buf) >= chunk_size:
            path = os.path.join(hdf5_folder, f"{chrom}_{chunk_idx}.h5")
            make_hdf5_file(path, np.stack(xs_buf), np.stack(ys_buf))
            print(f"Wrote {len(xs_buf)} samples to {path}")
            xs_buf, ys_buf = [], []
            chunk_idx += 1

    for _, row in df_tss.iterrows():
        pos = int(row["pos"]) - 1  # 0-based
        strand = row.get("strand", 1)

        seq_str = str(fasta[chrom][pos - half_in : pos + half_in]).upper()
        if len(seq_str) < input_length:
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

    for bw in bws_fwd + bws_rev:
        bw.close()
    print(f"Preprocessing complete for {chrom}: {chunk_idx} HDF5 files written.")


def main():
    """Parse args and run preprocess_chrom for the specified chromosome."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_folder", required=True)
    parser.add_argument("--tss_file", required=True)
    parser.add_argument("--fasta_file", required=True)
    parser.add_argument("--bigwig_files", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--input_length", type=int, required=True)
    parser.add_argument("--output_length", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, default=256)
    args = parser.parse_args()

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
