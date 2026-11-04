import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")
figwig = pytest.importorskip("figwig")

from promoterai_torch import preprocess
from promoterai_torch.dataset import onehot_encode


def _write_bigwig(path, chrom_sizes, chrom, values):
    """Write per-base values for one chrom (starting at 0) to a real BigWig file."""
    values = np.asarray(values, dtype="float32")[None, :]
    figwig.write_bigwig(str(path), chrom_sizes, chrom, [0], values, zooms=0)
    return str(path)


def test_preprocess_uses_stranded_bigwigs_xforms_and_reverses_minus(tmp_path):
    fasta_path = tmp_path / "mini.fa"
    fasta_path.write_text(">chr1\nACGTACGTACGT\n")
    tss_path = tmp_path / "tss.tsv"
    pd.DataFrame(
        [
            {"chrom": "chr1", "pos": 6, "strand": "+"},
            {"chrom": "chr1", "pos": 6, "strand": "-"},
        ]
    ).to_csv(tss_path, sep="\t", index=False)
    sizes = {"chr1": 12}
    fwd = _write_bigwig(
        tmp_path / "fwd.bw", sizes, "chr1", [0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0]
    )
    rev = _write_bigwig(
        tmp_path / "rev.bw", sizes, "chr1", [0, 0, 0, 10, 20, 30, 40, 0, 0, 0, 0, 0]
    )
    bw_path = tmp_path / "bigwigs.tsv"
    pd.DataFrame(
        [{"fwd": fwd, "rev": rev, "xform": "lambda x: x * 2"}]
    ).to_csv(bw_path, sep="\t", index=False)

    out_dir = tmp_path / "h5"
    preprocess.preprocess_chrom(
        tss_file=str(tss_path),
        fasta_file=str(fasta_path),
        bigwig_files_tsv=str(bw_path),
        chrom="chr1",
        hdf5_folder=str(out_dir),
        input_length=4,
        output_length=4,
        chunk_size=8,
    )

    with h5py.File(out_dir / "chr1.h5", "r") as handle:
        # One sample per HDF5 chunk: training reads random single rows, and a
        # multi-row chunk would be decompressed whole for each of them.
        assert handle["x"].chunks == (1, 4, 4)
        assert handle["y"].chunks == (1, 4, 1)
        x = handle["x"][:]
        y = handle["y"][:]

    plus_x = onehot_encode("TACG")
    np.testing.assert_array_equal(x[0], plus_x)
    np.testing.assert_array_equal(x[1], plus_x[::-1, ::-1])
    np.testing.assert_allclose(y[0, :, 0], np.array([2, 4, 6, 8], dtype="float32"))
    np.testing.assert_allclose(y[1, :, 0], np.array([80, 60, 40, 20], dtype="float32"))


def test_preprocess_keeps_boundary_tss_as_zero_row_matching_tf_generator(tmp_path):
    """A TSS too close to a chromosome edge for a full window must stay in the
    dataset as an all-zero row (matching Illumina's generator.py), not be
    dropped -- dropping it shrinks the dataset and skews per-species sizes
    used for multi-species sampling ratios.
    """
    fasta_path = tmp_path / "mini.fa"
    fasta_path.write_text(">chr1\nACGTACGTACGT\n")  # 12 bp
    tss_path = tmp_path / "tss.tsv"
    pd.DataFrame(
        [
            {"chrom": "chr1", "pos": 11},  # too close to the end for input_length=8
            {"chrom": "chr1", "pos": 6},  # comfortably interior
        ]
    ).to_csv(tss_path, sep="\t", index=False)
    bw = _write_bigwig(tmp_path / "track.bw", {"chr1": 12}, "chr1", np.arange(1, 13))
    bw_path = tmp_path / "bigwigs.tsv"
    # xform(0) != 0, so this also checks boundary rows bypass the transform.
    pd.DataFrame([{"fwd": bw, "rev": bw, "xform": "lambda x: x + 1"}]).to_csv(
        bw_path, sep="\t", index=False
    )

    out_dir = tmp_path / "h5"
    preprocess.preprocess_chrom(
        tss_file=str(tss_path),
        fasta_file=str(fasta_path),
        bigwig_files_tsv=str(bw_path),
        chrom="chr1",
        hdf5_folder=str(out_dir),
        input_length=8,
        output_length=4,
        chunk_size=8,
    )

    with h5py.File(out_dir / "chr1.h5", "r") as handle:
        x = handle["x"][:]
        y = handle["y"][:]

    assert x.shape[0] == 2  # both TSS rows kept, neither dropped
    assert x[0].sum() == 0.0
    assert y[0].sum() == 0.0
    assert x[1].sum() > 0.0
    assert y[1].sum() > 0.0


def test_preprocess_zero_fills_bigwig_window_past_chrom_end(tmp_path):
    """figwig returns NaN past a chromosome's end; preprocessing must store 0
    there (pybigtools' oob=0.0 behaviour), including when the BigWig's chrom
    is shorter than the FASTA's."""
    fasta_path = tmp_path / "mini.fa"
    fasta_path.write_text(">chr1\nACGTACGTACGT\n")  # 12 bp
    tss_path = tmp_path / "tss.tsv"
    pd.DataFrame([{"chrom": "chr1", "pos": 7}]).to_csv(tss_path, sep="\t", index=False)
    bw = _write_bigwig(tmp_path / "short.bw", {"chr1": 8}, "chr1", np.arange(1, 9))
    bw_path = tmp_path / "bigwigs.tsv"
    pd.DataFrame([{"fwd": bw, "rev": bw, "xform": "np.arcsinh"}]).to_csv(
        bw_path, sep="\t", index=False
    )

    out_dir = tmp_path / "h5"
    preprocess.preprocess_chrom(
        tss_file=str(tss_path),
        fasta_file=str(fasta_path),
        bigwig_files_tsv=str(bw_path),
        chrom="chr1",
        hdf5_folder=str(out_dir),
        input_length=8,
        output_length=6,  # window [3, 9) runs 1 bp past the BigWig's chr1
        chunk_size=8,
    )

    with h5py.File(out_dir / "chr1.h5", "r") as handle:
        y = handle["y"][:]

    expected = np.arcsinh(np.array([4, 5, 6, 7, 8, 0], dtype="float32"))
    np.testing.assert_allclose(y[0, :, 0], expected, rtol=1e-6)
    assert np.isfinite(y).all()


def test_preprocess_reports_missing_figwig(tmp_path, monkeypatch):
    fasta_path = tmp_path / "mini.fa"
    fasta_path.write_text(">chr1\nACGT\n")
    tss_path = tmp_path / "tss.tsv"
    pd.DataFrame([{"chrom": "chr1", "pos": 2}]).to_csv(
        tss_path, sep="\t", index=False
    )
    bw_path = tmp_path / "bigwigs.tsv"
    pd.DataFrame([{"fwd": "fwd.bw", "rev": "rev.bw", "xform": "lambda x: x"}]).to_csv(
        bw_path, sep="\t", index=False
    )

    import_error = ModuleNotFoundError("No module named 'figwig'")
    monkeypatch.setattr(preprocess, "figwig", None)
    monkeypatch.setattr(preprocess, "_figwig_import_error", import_error)

    with np.testing.assert_raises_regex(
        ImportError, r'pip install "promoterai-torch\[train\]"'
    ):
        preprocess.preprocess_chrom(
            tss_file=str(tss_path),
            fasta_file=str(fasta_path),
            bigwig_files_tsv=str(bw_path),
            chrom="chr1",
            hdf5_folder=str(tmp_path / "h5"),
            input_length=4,
            output_length=4,
            chunk_size=1,
        )
