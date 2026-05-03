import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(EXAMPLES))

from compare_tf_torch_tracks_promoters import (  # noqa: E402
    Promoter,
    parse_promoter,
    sequence_for_promoter,
)
from compare_tf_torch_tracks_random import random_onehot  # noqa: E402
from track_parity_utils import (  # noqa: E402
    compare_track_outputs,
    normalize_tf_outputs,
)


def test_normalize_tf_outputs_orders_multi_species_dict():
    outputs = {
        "shortcut_mouse": np.full((1, 2, 3), 2.0, dtype=np.float32),
        "shortcut_human": np.full((1, 2, 4), 1.0, dtype=np.float32),
    }

    normalized = normalize_tf_outputs(outputs)

    assert len(normalized) == 2
    assert normalized[0].shape == (1, 2, 4)
    assert normalized[1].shape == (1, 2, 3)
    assert np.all(normalized[0] == 1.0)
    assert np.all(normalized[1] == 2.0)


def test_compare_track_outputs_reports_per_sample_and_head_errors():
    tf_outputs = [
        np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32),
        np.array([[[10.0]], [[20.0]]], dtype=np.float32),
    ]
    torch_outputs = [
        np.array([[[1.5], [1.0]], [[3.0], [5.0]]], dtype=np.float32),
        np.array([[[9.0]], [[23.0]]], dtype=np.float32),
    ]

    results = compare_track_outputs(["a", "b"], tf_outputs, torch_outputs)

    assert [(r.sample, r.head, r.shape) for r in results] == [
        ("a", 0, (2, 1)),
        ("b", 0, (2, 1)),
        ("a", 1, (1, 1)),
        ("b", 1, (1, 1)),
    ]
    assert results[0].mean_abs_error == 0.75
    assert results[0].max_abs_error == 1.0
    assert results[3].mean_abs_error == 3.0
    assert results[3].max_abs_error == 3.0


def test_random_onehot_is_deterministic_and_valid():
    x1 = random_onehot(n_sequences=3, input_length=8, seed=17)
    x2 = random_onehot(n_sequences=3, input_length=8, seed=17)

    np.testing.assert_array_equal(x1, x2)
    assert x1.shape == (3, 8, 4)
    np.testing.assert_array_equal(x1.sum(axis=-1), np.ones((3, 8), dtype=np.float32))


def test_promoter_sequence_extraction_pads_and_reverse_complements(tmp_path):
    import pyfaidx

    fasta_path = tmp_path / "mini.fa"
    fasta_path.write_text(">chr1\nACGTACGT\n")
    fasta = pyfaidx.Fasta(str(fasta_path))

    plus = sequence_for_promoter(
        fasta,
        Promoter(name="plus", chrom="chr1", tss=2, strand="+"),
        input_length=6,
    )
    minus = sequence_for_promoter(
        fasta,
        Promoter(name="minus", chrom="chr1", tss=2, strand="-"),
        input_length=6,
    )

    assert plus.shape == (6, 4)
    np.testing.assert_array_equal(minus, plus[::-1, ::-1])
    assert plus[0].sum() == 0.0  # left-padded N


def test_parse_promoter_spec():
    promoter = parse_promoter("TERT:chr5:1294988:-")

    assert promoter == Promoter(name="TERT", chrom="chr5", tss=1294988, strand="-")


def test_track_parity_scripts_expose_cli_help():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(EXAMPLES)])

    for script in (
        "compare_tf_torch_tracks_random.py",
        "compare_tf_torch_tracks_promoters.py",
    ):
        result = subprocess.run(
            [sys.executable, str(EXAMPLES / script), "--help"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        assert "--tf_device" in result.stdout
        assert "--interleaved" in result.stdout
