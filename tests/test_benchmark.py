import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from promoterai_torch.benchmark import auroc, benchmark_aurocs
from promoterai_torch.benchmark import benchmark_paths
from promoterai_torch.benchmark import score_benchmark_file


ROOT = Path(__file__).resolve().parents[1]


def test_auroc_perfect_reversed_and_ties():
    y_true = np.array([0, 0, 1, 1])

    assert auroc(y_true, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(y_true, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auroc(y_true, np.array([1.0, 1.0, 1.0, 1.0])) == 0.5


def test_auroc_returns_nan_for_one_class():
    value = auroc(np.array([1, 1]), np.array([0.2, 0.8]))

    assert math.isnan(value)


def test_benchmark_aurocs_use_signed_directions():
    df = pd.DataFrame(
        {
            "consequence": ["under", "under", "none", "none", "over", "over"],
            "score": [-0.9, -0.7, -0.1, 0.1, 0.7, 0.9],
        }
    )

    metrics = benchmark_aurocs(df)

    assert metrics["under_over_auroc"] == 1.0
    assert metrics["under_null_auroc"] == 1.0
    assert metrics["over_null_auroc"] == 1.0
    assert metrics["n_under"] == 2
    assert metrics["n_over"] == 2
    assert metrics["n_null"] == 2


def test_benchmark_paths_can_select_named_datasets(tmp_path):
    for name in ("GTEx_outlier.tsv", "MPRA_eQTL.tsv", "notes.txt"):
        (tmp_path / name).write_text("x\n")

    paths = benchmark_paths(tmp_path, ["GTEx_outlier", "MPRA_eQTL.tsv"])

    assert [path.name for path in paths] == ["GTEx_outlier.tsv", "MPRA_eQTL.tsv"]


def test_benchmark_paths_defaults_to_all_tsvs(tmp_path):
    for name in ("b.tsv", "a.tsv", "notes.txt"):
        (tmp_path / name).write_text("x\n")

    paths = benchmark_paths(tmp_path)

    assert [path.name for path in paths] == ["a.tsv", "b.tsv"]


def test_score_benchmark_file_allows_hg38_only(tmp_path, monkeypatch):
    benchmark_tsv = tmp_path / "variants.tsv"
    pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "pos": [10, 20],
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            "consequence": ["under", "over"],
        }
    ).to_csv(benchmark_tsv, sep="\t", index=False)

    calls = []

    def fake_score_variants(df, *, model_checkpoint, **kwargs):
        calls.append(model_checkpoint)
        return np.array([0.2, -0.4])

    monkeypatch.setattr("promoterai_torch.benchmark.score_variants", fake_score_variants)

    scored = score_benchmark_file(
        benchmark_tsv,
        hg38_finetune_checkpoint="hg38.pt",
        fasta_file="hg38.fa",
        input_length=20480,
    )

    assert calls == ["hg38.pt"]
    assert "hg38_finetune_score" in scored.columns
    assert "hg38_mm10_finetune_score" not in scored.columns
    np.testing.assert_allclose(scored["score"], [0.2, -0.4])


def test_score_benchmark_file_ensembles_when_hg38_mm10_is_provided(
    tmp_path, monkeypatch
):
    benchmark_tsv = tmp_path / "variants.tsv"
    pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "pos": [10, 20],
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            "consequence": ["under", "over"],
        }
    ).to_csv(benchmark_tsv, sep="\t", index=False)

    scores_by_checkpoint = {
        "hg38.pt": np.array([0.2, -0.4]),
        "hg38_mm10.pt": np.array([0.6, 0.2]),
    }

    def fake_score_variants(df, *, model_checkpoint, **kwargs):
        return scores_by_checkpoint[model_checkpoint]

    monkeypatch.setattr("promoterai_torch.benchmark.score_variants", fake_score_variants)

    scored = score_benchmark_file(
        benchmark_tsv,
        hg38_finetune_checkpoint="hg38.pt",
        hg38_mm10_finetune_checkpoint="hg38_mm10.pt",
        fasta_file="hg38.fa",
        input_length=20480,
    )

    assert "hg38_mm10_finetune_score" in scored.columns
    np.testing.assert_allclose(scored["score"], [0.4, -0.1])


def test_tensorflow_benchmark_script_exposes_cli_help():
    script = ROOT / "examples" / "paper_benchmark" / "benchmark_variant_scores_tf.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--hg38_finetune_model_folder" in result.stdout
    assert "--devices" in result.stdout


def test_tensorflow_benchmark_script_is_standalone_from_promoterai_torch():
    script = ROOT / "examples" / "paper_benchmark" / "benchmark_variant_scores_tf.py"

    assert "promoterai_torch" not in script.read_text()
