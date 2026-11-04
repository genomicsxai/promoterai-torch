"""Regenerate tests/data/toy_savedmodel/ and its reference weights (needs the `tf` group).

A tiny random-weight two-species (mouse head first) model in the Illumina weight
naming convention, used by tests/test_savedmodel.py to check the TF-free
SavedModel reader without TensorFlow installed. Not an Illumina model.

    uv sync --group dev --group tf
    uv run python tests/data/make_toy_savedmodel.py
"""

import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import tf_keras  # noqa: E402

from tests.test_convert import _build_tf_keras_model  # noqa: E402

out = os.path.join(HERE, "toy_savedmodel")
shutil.rmtree(out, ignore_errors=True)
tf_keras.utils.set_random_seed(0)
model = _build_tf_keras_model(
    num_blocks=2,
    model_dim=4,
    output_dims=(3, 2),
    shortcut_layer_freq=1,
    species=("mouse", "human"),
)
# Saved with traces, like the published Illumina SavedModels: that is what
# writes the root's ordered `variables` list the reader relies on.
tf_keras.models.save_model(model, out, save_format="tf")

reloaded = tf_keras.models.load_model(out)
weights = {v.name.removesuffix(":0"): v.numpy() for v in reloaded.weights}
np.savez(
    os.path.join(HERE, "toy_savedmodel_weights.npz"),
    **{f"w{i}": v for i, v in enumerate(weights.values())},
)
with open(os.path.join(HERE, "toy_savedmodel_weights.json"), "w") as handle:
    json.dump(list(weights), handle, indent=1)
print(f"Wrote {out} ({len(weights)} variables)")
