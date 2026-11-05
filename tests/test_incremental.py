"""Receptive-field-limited substitution scoring must match TwinModel's full passes."""

import argparse

import numpy as np
import pandas as pd
import pytest
import torch

from promoterai_torch import score
from promoterai_torch.architecture import PromoterAI, TwinModel
from promoterai_torch.incremental import edit_span, reference_layers, substitution_diffs


def _model(num_blocks=8, model_dim=8, output_dims=(5, 3), output_crop=0, seed=0):
    torch.manual_seed(seed)
    model = PromoterAI(
        num_blocks=num_blocks,
        model_dim=model_dim,
        output_dims=list(output_dims),
        output_crop=output_crop,
    )
    # Non-trivial BatchNorm running stats and larger weights than the init, so
    # the ReLUs, residuals and BN affine terms all actually matter.
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.2, 0.2)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 0.3)
            m.bias.data.normal_(0, 0.1)
    return model.double().eval()


def _onehot(idx):
    return torch.eye(4, dtype=torch.float64)[idx]


def _alts(x_ref, lo, bases):
    """(B, L, 4) alts substituting bases[k] (a list of base indices) at lo.."""
    alts = x_ref.repeat(len(bases), 1, 1)
    for k, sub in enumerate(bases):
        alts[k, lo : lo + len(sub)] = _onehot(torch.tensor(sub))
    return alts


@pytest.mark.parametrize("output_crop", [0, 192])
@pytest.mark.parametrize(
    "lo, bases",
    [
        (128, [[0], [1], [2], [3]]),  # SNVs at the center (one is a no-op alt == ref)
        (1, [[2], [3]]),  # next to the start: windows clipped at the left edge
        (254, [[0], [1]]),  # next to the end: windows clipped at the right edge
        (100, [[0, 1, 2], [3, 3, 3]]),  # MNVs
    ],
)
def test_substitution_diffs_match_twin_model(output_crop, lo, bases):
    L = 256
    model = _model(output_crop=output_crop)
    g = torch.Generator().manual_seed(1)
    x_ref = _onehot(torch.randint(0, 4, (1, L), generator=g))
    x_alt = _alts(x_ref, lo, bases)

    expected = TwinModel(model).eval()(x_ref.expand(len(bases), -1, -1), x_alt)
    span = edit_span(x_ref, x_alt)
    got = substitution_diffs(model, reference_layers(model, x_ref), x_alt, span)

    torch.testing.assert_close(got, expected, rtol=1e-9, atol=1e-12)


def test_substitution_diffs_real_scale_dilations():
    """24 blocks (dilations up to 1024) on a length where late windows cover nearly all of it."""
    L = 4096
    model = _model(num_blocks=24, model_dim=4, output_dims=(3,), output_crop=L - 1024)
    x_ref = _onehot(torch.randint(0, 4, (1, L), generator=torch.Generator().manual_seed(2)))
    x_alt = _alts(x_ref, L // 2, [[0], [1], [2]])
    expected = TwinModel(model).eval()(x_ref.expand(3, -1, -1), x_alt)
    got = substitution_diffs(model, reference_layers(model, x_ref), x_alt, edit_span(x_ref, x_alt))
    torch.testing.assert_close(got, expected, rtol=1e-9, atol=1e-12)


def test_edit_span():
    x = _onehot(torch.tensor([[0, 1, 2, 3, 0, 1]]))
    assert edit_span(x, x.clone()) is None
    y = x.clone()
    y[0, 2] = _onehot(torch.tensor(0))
    y[0, 4] = _onehot(torch.tensor(3))
    assert edit_span(x, y) == (2, 5)


def _run_score(tmp_path, ckpt, var_path, fasta_path, incremental):
    out = tmp_path / f"out_{incremental}.tsv"
    args = score.build_parser().parse_args(
        [
            "--model_checkpoint", str(ckpt),
            "--var_file", str(var_path),
            "--fasta_file", str(fasta_path),
            "--input_length", "256",
            "--device", "cpu",
            "--num_workers", "0",
            "--output", str(out),
            "--incremental" if incremental else "--no-incremental",
        ]
    )
    score.main(args)
    return pd.read_csv(out, sep="\t")


def test_score_cli_incremental_matches_full_passes(tmp_path):
    rng = np.random.default_rng(0)
    L_chrom = 2000
    seq = "".join(rng.choice(list("ACGT"), L_chrom))
    fasta_path = tmp_path / "g.fa"
    fasta_path.write_text(">chr1\n" + seq + "\n")

    model = _model(output_crop=192).float()
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
    for pos in (500, 900):  # all three SNVs at two sites, both strands
        ref = seq[pos - 1]
        for alt in "ACGT".replace(ref, ""):
            for strand in (1, -1):
                rows.append((pos, ref, alt, strand))
    rows += [
        (1200, seq[1199:1202], "TTT" if seq[1199:1202] != "TTT" else "AAA", 1),  # MNV
        (1300, "A" if seq[1299] != "A" else "C", "G", 1),  # ref mismatch -> placeholder
        (1400, seq[1399], seq[1399] + "AC", 1),  # insertion -> full pass
        (1500, seq[1499:1502], seq[1499], -1),  # deletion -> full pass
        (60, seq[59], "ACGT".replace(seq[59], "")[0], 1),  # near the chromosome start (N-padded)
    ]
    var_path = tmp_path / "variants.tsv"
    pd.DataFrame(rows, columns=["pos", "ref", "alt", "strand"]).assign(chrom="chr1")[
        ["chrom", "pos", "ref", "alt", "strand"]
    ].to_csv(var_path, sep="\t", index=False)

    full = _run_score(tmp_path, ckpt, var_path, fasta_path, incremental=False)
    fast = _run_score(tmp_path, ckpt, var_path, fasta_path, incremental=True)

    pd.testing.assert_frame_equal(full.drop(columns="score"), fast.drop(columns="score"))
    np.testing.assert_allclose(fast["score"], full["score"], rtol=0, atol=2e-4)
    assert (fast["score"] == full["score"]).mean() > 0.9  # same after round(4) except rare ties
    assert fast.loc[len(rows) - 4, "score"] == 0.0  # ref mismatch scores 0, as before


def test_site_groups_split_substitutions_from_indels():
    df = pd.DataFrame(
        {
            "chrom": ["chr1"] * 5,
            "pos": [10, 10, 10, 20, 30],
            "ref": ["A", "A", "A", "AC", "G"],
            "alt": ["C", "G", "C", "A", "T"],
            "strand": [1, 1, -1, 1, 1],
        }
    )
    groups, other = score._site_groups(df)
    assert sorted(map(sorted, groups)) == [[0, 1], [2], [4]]
    assert other.tolist() == [3]


def test_incremental_flag_defaults_on():
    args = score.build_parser().parse_args(
        ["--model_checkpoint", "m.pt", "--var_file", "v.tsv", "--fasta_file", "g.fa"]
    )
    assert args.incremental is True
    assert isinstance(score.build_parser(argparse.ArgumentParser()), argparse.ArgumentParser)
