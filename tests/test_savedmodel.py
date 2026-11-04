"""
Tests for the TF-free SavedModel reader (promoterai_torch.savedmodel).

The fixture-based tests need no TensorFlow: tests/data/toy_savedmodel is a tiny
random-weight two-species model (mouse head first), with reference weights taken
from tf_keras when it was written (regenerate with
tests/data/make_toy_savedmodel.py). The tf_keras comparison tests and the
real-checkpoint test run only when tf-keras is installed (the `tf` dependency
group).
"""

import json
import os
import re

import numpy as np
import pytest
import torch

from promoterai_torch.savedmodel import read_savedmodel_variables

DATA = os.path.join(os.path.dirname(__file__), "data")
TOY = os.path.join(DATA, "toy_savedmodel")


def _toy_reference():
    with open(os.path.join(DATA, "toy_savedmodel_weights.json")) as handle:
        names = json.load(handle)
    arrays = np.load(os.path.join(DATA, "toy_savedmodel_weights.npz"))
    return {name: arrays[f"w{i}"] for i, name in enumerate(names)}


def _species_order(names):
    """First-appearance order of shortcut_{species}{N} heads (as convert_tf_weights uses)."""
    order = []
    for name in names:
        m = re.match(r"^shortcut(?:_([a-zA-Z]+))?\d+/kernel$", name)
        if m and (m.group(1) or "") not in order:
            order.append(m.group(1) or "")
    return order


def _assert_same_weights(got, ref):
    assert list(got) == list(ref)  # same variables, in keras_model.weights order
    for name, value in ref.items():
        assert got[name].dtype == value.dtype, name
        np.testing.assert_array_equal(got[name], value, err_msg=name)
    assert _species_order(got) == _species_order(ref)


# ── no TensorFlow needed ─────────────────────────────────────────────────────


def test_reader_matches_tf_keras_reference_without_tensorflow():
    got = read_savedmodel_variables(TOY)
    ref = _toy_reference()
    _assert_same_weights(got, ref)
    assert _species_order(got) == ["mouse", "human"]


def test_convert_tf_weights_from_toy_savedmodel(tmp_path):
    from promoterai_torch.utils import convert_tf_weights, load_pretrained

    out = str(tmp_path / "toy.pt")
    convert_tf_weights(TOY, out, input_length=64, output_length=64)
    model, args = load_pretrained(out)

    assert args["num_blocks"] == 2
    assert args["model_dim"] == 4
    assert args["species_order"] == ["mouse", "human"]
    assert args["output_dims"] == [3, 2]

    ref = _toy_reference()
    sd = model.state_dict()
    np.testing.assert_array_equal(
        sd["stem.weight"].numpy(), ref["dense/kernel"].T[:, :, None]
    )
    np.testing.assert_array_equal(sd["stem.bias"].numpy(), ref["dense/bias"])

    model.eval()
    with torch.no_grad():
        out_tracks = model(torch.eye(4)[torch.randint(0, 4, (1, 64))])
    assert [o.shape[-1] for o in out_tracks] == [3, 2]


def test_reader_rejects_non_savedmodel_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a SavedModel"):
        read_savedmodel_variables(tmp_path)


def test_reader_rejects_corrupt_index(tmp_path):
    import shutil

    bad = tmp_path / "bad"
    shutil.copytree(TOY, bad)
    (bad / "variables" / "variables.index").write_bytes(b"not an sstable" * 8)
    with pytest.raises(ValueError, match="tensor-bundle index"):
        read_savedmodel_variables(bad)


# ── cross-checks against tf_keras ────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(output_dims=(8,)),
        dict(output_dims=(6, 8), species=("mouse", "human")),
        dict(output_dims=(8, 6), species=("human", "mouse")),
    ],
    ids=["single", "multi-mouse-first", "multi-human-first"],
)
def test_reader_matches_tf_keras_load_model(tmp_path, kwargs):
    tf_keras = pytest.importorskip("tf_keras", reason="tf-keras not installed")
    from tests.test_convert import _build_tf_keras_model, _save_savedmodel

    path = str(tmp_path / "keras_model")
    _save_savedmodel(_build_tf_keras_model(num_blocks=8, model_dim=16, **kwargs), path)
    reloaded = tf_keras.models.load_model(path)
    ref = {v.name.removesuffix(":0"): v.numpy() for v in reloaded.weights}

    _assert_same_weights(read_savedmodel_variables(path), ref)


def test_reader_matches_tf_keras_on_real_savedmodel(keras_savedmodel_path):
    """Run with --keras-savedmodel-path on each licensed Illumina SavedModel."""
    tf_keras = pytest.importorskip("tf_keras", reason="tf-keras not installed")

    reloaded = tf_keras.models.load_model(keras_savedmodel_path)
    ref = {v.name.removesuffix(":0"): v.numpy() for v in reloaded.weights}

    _assert_same_weights(read_savedmodel_variables(keras_savedmodel_path), ref)
