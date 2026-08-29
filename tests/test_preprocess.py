import types

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")

from promoterai_torch import preprocess
from promoterai_torch.dataset import onehot_encode


class FakeBigWig:
    def __init__(self, values):
        self.values_array = np.asarray(values, dtype="float32")
        self.closed = False

    def values(self, chrom, start, end, missing=0.0, oob=0.0):
        return self.values_array[: end - start]

    def close(self):
        self.closed = True


def test_preprocess_uses_stranded_bigwigs_xforms_and_reverses_minus(tmp_path, monkeypatch):
    fasta_path = tmp_path / "mini.fa"
    fasta_path.write_text(">chr1\nACGTACGTACGT\n")
    tss_path = tmp_path / "tss.tsv"
    pd.DataFrame(
        [
            {"chrom": "chr1", "pos": 6, "strand": "+"},
            {"chrom": "chr1", "pos": 6, "strand": "-"},
        ]
    ).to_csv(tss_path, sep="\t", index=False)
    bw_path = tmp_path / "bigwigs.tsv"
    pd.DataFrame(
        [{"fwd": "fwd.bw", "rev": "rev.bw", "xform": "lambda x: x * 2"}]
    ).to_csv(bw_path, sep="\t", index=False)

    opened = {}

    def fake_open(path):
        values = [1, 2, 3, 4] if path == "fwd.bw" else [10, 20, 30, 40]
        opened[path] = FakeBigWig(values)
        return opened[path]

    monkeypatch.setattr(preprocess, "pybigtools", types.SimpleNamespace(open=fake_open))

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
        x = handle["x"][:]
        y = handle["y"][:]

    plus_x = onehot_encode("TACG")
    np.testing.assert_array_equal(x[0], plus_x)
    np.testing.assert_array_equal(x[1], plus_x[::-1, ::-1])
    np.testing.assert_allclose(y[0, :, 0], np.array([2, 4, 6, 8], dtype="float32"))
    np.testing.assert_allclose(y[1, :, 0], np.array([80, 60, 40, 20], dtype="float32"))
    assert opened["fwd.bw"].closed
    assert opened["rev.bw"].closed


def test_preprocess_keeps_boundary_tss_as_zero_row_matching_tf_generator(
    tmp_path, monkeypatch
):
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
    bw_path = tmp_path / "bigwigs.tsv"
    pd.DataFrame(
        [{"fwd": "fwd.bw", "rev": "rev.bw", "xform": "lambda x: x"}]
    ).to_csv(bw_path, sep="\t", index=False)

    monkeypatch.setattr(
        preprocess,
        "pybigtools",
        types.SimpleNamespace(open=lambda path: FakeBigWig([1, 2, 3, 4])),
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


def test_preprocess_reports_missing_pybigtools(tmp_path, monkeypatch):
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

    import_error = ModuleNotFoundError("No module named 'pybigtools'")
    monkeypatch.setattr(preprocess, "pybigtools", None)
    monkeypatch.setattr(preprocess, "_pybigtools_import_error", import_error)

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
