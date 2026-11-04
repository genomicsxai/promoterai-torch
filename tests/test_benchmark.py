import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from promoterai_torch.benchmark import (
    auroc,
    benchmark_aurocs,
    benchmark_paths,
    score_benchmark_file,
)

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


def test_site_shards_keep_sites_together():
    from promoterai_torch.benchmark import site_shards

    df = pd.DataFrame(
        {
            "chrom": ["chr1"] * 7 + ["chr2"] * 3,
            "pos": [10, 10, 10, 20, 30, 30, 40, 10, 10, 50],
            "strand": [1, 1, 1, 1, 1, -1, 1, 1, 1, 1],
        }
    )
    shards = site_shards(df, 3)
    assert len(shards) == 3
    assert sorted(np.concatenate(shards).tolist()) == list(range(len(df)))
    keys = list(zip(df.chrom, df.pos, df.strand))
    owner = {}
    for s, rows in enumerate(shards):
        for i in rows:
            assert owner.setdefault(keys[i], s) == s  # a site never spans shards
    assert all(len(s) for s in shards)
    assert [len(s) for s in site_shards(df, 1)] == [len(df)]
    assert [len(s) for s in site_shards(df.iloc[:1], 2)] == [1, 0]


def _toy_benchmark(tmp_path):
    """A toy checkpoint, genome and benchmark TSV (SNV sites with 1-3 alts, both strands)."""
    import torch

    from promoterai_torch.architecture import PromoterAI

    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), 2000))
    fasta = tmp_path / "g.fa"
    fasta.write_text(">chr1\n" + seq + "\n")
    torch.manual_seed(0)
    model = PromoterAI(num_blocks=8, model_dim=8, output_dims=[5, 3], output_crop=192)
    for m in model.modules():  # weights large enough that the scores aren't ~0
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 0.3)
            m.bias.data.normal_(0, 0.1)
    ckpt = tmp_path / "toy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": {
                "num_blocks": 8, "model_dim": 8, "output_dims": [5, 3],
                "output_crop": 192, "input_length": 256, "output_length": 64,
            },
        },
        ckpt,
    )
    rows = []
    for n_alts, pos, strand in ((3, 500, 1), (1, 700, -1), (2, 900, 1), (1, 1100, 1), (3, 1300, -1)):
        ref = seq[pos - 1]
        for alt in "ACGT".replace(ref, "")[:n_alts]:
            rows.append(("chr1", pos, ref, alt, strand, "g", ["under", "none", "over"][len(rows) % 3]))
    bench = tmp_path / "bench"
    bench.mkdir()
    pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt", "strand", "gene", "consequence"]).to_csv(
        bench / "toy.tsv", sep="\t", index=False
    )
    return ckpt, fasta, bench


def test_benchmark_incremental_matches_full_passes(tmp_path):
    from promoterai_torch.benchmark import run_benchmarks

    ckpt, fasta, bench = _toy_benchmark(tmp_path)
    common = dict(
        hg38_finetune_checkpoint=ckpt, hg38_mm10_finetune_checkpoint=ckpt, fasta_file=fasta,
        input_length=256, device="cpu", num_workers=0,
    )
    run_benchmarks(bench, output_dir=tmp_path / "full", incremental=False, **common)
    run_benchmarks(bench, output_dir=tmp_path / "fast", **common)  # incremental by default
    full = pd.read_csv(tmp_path / "full" / "toy.scores.tsv", sep="\t")
    fast = pd.read_csv(tmp_path / "fast" / "toy.scores.tsv", sep="\t")
    assert full["score"].abs().max() > 0.01
    for col in ("hg38_finetune_score", "hg38_mm10_finetune_score", "score"):
        np.testing.assert_allclose(fast[col], full[col], rtol=0, atol=2e-4)
    assert (fast["score"] == full["score"]).mean() > 0.9  # same after round(4) except rare ties


def test_benchmark_multi_device_incremental_matches_single_device(tmp_path):
    from promoterai_torch.benchmark import score_benchmark_file, score_benchmark_file_multi_device

    ckpt, fasta, bench = _toy_benchmark(tmp_path)
    common = dict(hg38_finetune_checkpoint=ckpt, fasta_file=fasta, input_length=256, num_workers=0)
    single = score_benchmark_file(bench / "toy.tsv", device="cpu", **common)
    multi = score_benchmark_file_multi_device(bench / "toy.tsv", devices=["cpu", "cpu"], **common)
    pd.testing.assert_frame_equal(multi, single, check_dtype=False)  # multi-device collects into float64
