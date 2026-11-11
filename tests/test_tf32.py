"""--tf32 / set_tf32: the CLI commands match TensorFlow's default TF32 behaviour."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from promoterai_torch import finetune, score, train
from promoterai_torch.benchmark import score_benchmark_file
from promoterai_torch.utils import set_tf32

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def restore_tf32():
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def test_set_tf32_sets_matmul_and_cudnn(restore_tf32):
    set_tf32(True)
    assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    set_tf32(False)
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32


@pytest.mark.parametrize("module", [score, train, finetune])
def test_cli_tf32_flag_defaults_on(module):
    parser = module.build_parser()
    assert parser.get_default("tf32") is True
    (action,) = [a for a in parser._actions if a.dest == "tf32"]
    assert "--no-tf32" in action.option_strings


class _Stop(Exception):
    pass


@pytest.mark.parametrize("flag, expected", [([], True), (["--no-tf32"], False)])
def test_score_main_applies_tf32_first(monkeypatch, flag, expected):
    seen = []

    def fake_set_tf32(enabled):
        seen.append(enabled)
        raise _Stop  # stop before loading a checkpoint

    monkeypatch.setattr(score, "set_tf32", fake_set_tf32)
    args = score.build_parser().parse_args(
        ["--model_checkpoint", "m.pt", "--var_file", "v.tsv", "--fasta_file", "f.fa", *flag]
    )
    with pytest.raises(_Stop):
        score.main(args)
    assert seen == [expected]


def test_benchmark_passes_tf32_through(tmp_path, monkeypatch):
    benchmark_tsv = tmp_path / "variants.tsv"
    pd.DataFrame(
        {"chrom": ["chr1"], "pos": [10], "ref": ["A"], "alt": ["G"], "consequence": ["under"]}
    ).to_csv(benchmark_tsv, sep="\t", index=False)
    seen = []

    def fake_score_variants(df, *, model_checkpoint, tf32=None, **kwargs):
        seen.append(tf32)
        return np.array([0.1])

    monkeypatch.setattr("promoterai_torch.benchmark.score_variants", fake_score_variants)
    for tf32 in (None, True, False):
        score_benchmark_file(
            benchmark_tsv,
            hg38_finetune_checkpoint="hg38.pt",
            fasta_file="hg38.fa",
            input_length=20480,
            tf32=tf32,
        )
    assert seen == [None, True, False]


def test_paper_benchmark_script_exposes_tf32_flag():
    script = ROOT / "examples" / "paper_benchmark" / "benchmark_variant_scores.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], check=True, capture_output=True, text=True
    )
    assert "--no-tf32" in result.stdout
