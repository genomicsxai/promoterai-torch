"""
Cross-framework single-step training equivalence on a REAL checkpoint (GPU).

Runs the same comparison as tests/test_tf_gradient_equivalence.py -- one
AdamW(clipnorm=...) step in Keras and PyTorch through identical (converted)
weights -- but against a real, full-scale Illumina PromoterAI Keras
SavedModel (published: num_blocks=24, model_dim=1024) rather than the toy
8-block/dim-16 model used there. That toy test only proves the mechanics are
right at a scale that runs in seconds on a laptop; it doesn't exercise
anything that only shows up at full depth (24 stacked MetaFormer blocks
accumulate more floating-point divergence between frameworks) or full width.

Requires:
- a real Illumina PromoterAI Keras SavedModel, which is licensed and not
  distributed with this repo, passed via --keras-savedmodel-path
- enough memory to hold and backprop through the full model -- practically a
  GPU, hence the separate --device/--gradient-batch-size options

Skipped automatically unless --keras-savedmodel-path is given (including in
CI, where the licensed model is never available).

    uv sync --group dev --extra convert
    uv run pytest tests/test_tf_gradient_equivalence_real.py -v -s \\
        --keras-savedmodel-path /path/to/promoterai_keras_model \\
        --device cuda --gradient-batch-size 2

Inputs are random one-hot sequences and random targets, not real genomic
data -- this checks optimizer/gradient *mechanics* at real scale, not model
predictions (those are validated separately in
tests/test_track_parity_examples.py and examples/, using real sequences and
the model's real predictions).

Thresholds are looser than tests/test_tf_gradient_equivalence.py's: gradients
accumulate more floating-point divergence through 24 stacked blocks than
through 8, per alphagenome-pytorch's tests/README.md tolerance notes.
"""

import numpy as np
import pytest

from tests.gradient_comparison_utils import assert_pass_rate
from tests.keras_pytorch_step import force_tf_cpu, run_single_step

pytest.importorskip("tf_keras", reason="tf-keras not installed")


def test_real_checkpoint_single_step_gradient_equivalence(
    tmp_path,
    keras_savedmodel_path,
    gradient_device,
    gradient_batch_size,
    gradient_input_length,
    gradient_output_length,
):
    """One AdamW(clipnorm=...) step on identical weights/batch, real checkpoint scale."""
    force_tf_cpu()  # before any TF op, including inside convert_tf_weights below
    import tf_keras as keras

    from promoterai_torch.utils import convert_tf_weights, load_pretrained

    out_pt = str(tmp_path / "model.pt")
    convert_tf_weights(
        keras_savedmodel_path,
        out_pt,
        input_length=gradient_input_length,
        output_length=gradient_output_length,
    )
    pt_model, args = load_pretrained(out_pt)
    keras_model = keras.models.load_model(keras_savedmodel_path)

    num_blocks = args["num_blocks"]
    shortcut_layer_freq = args.get("shortcut_layer_freq", 4)
    shortcut_nums_desc = list(range(num_blocks, 0, -shortcut_layer_freq))
    output_dim = args["output_dims"][0]

    rng = np.random.default_rng(0)
    idx = rng.integers(0, 4, size=(gradient_batch_size, gradient_input_length))
    x_np = np.eye(4, dtype="float32")[idx]  # (B, L, 4)
    y_np = rng.normal(
        size=(gradient_batch_size, gradient_output_length, output_dim)
    ).astype("float32")

    lr, wd, eps, clip_norm = 5e-4, 5e-6, 1e-7, 1e-4
    results = run_single_step(
        keras_model,
        pt_model,
        x_np,
        y_np,
        num_blocks=num_blocks,
        shortcut_nums_desc=shortcut_nums_desc,
        lr=lr,
        wd=wd,
        eps=eps,
        clip_norm=clip_norm,
        device=gradient_device,
    )

    # --- 1. Forward loss ---
    np.testing.assert_allclose(
        results["loss_pt"], results["loss_keras"], atol=1e-3, rtol=1e-3,
        err_msg="forward-pass MSE loss differs between PyTorch and Keras",
    )

    # --- 2/3. Raw and post-clip gradients ---
    assert_pass_rate(
        results["raw_grad"], cosine_threshold=0.99, rel_l2_tol=5e-2,
        min_pass_rate=0.95, label="raw gradients",
    )
    assert_pass_rate(
        results["clipped_grad"], cosine_threshold=0.99, rel_l2_tol=5e-2,
        min_pass_rate=0.95, label="post-clip gradients",
    )

    # --- 4. AdamW parameter deltas ---
    #
    # See tests/test_tf_gradient_equivalence.py and notes/implementation.md for why
    # the real epsilon=1e-7 case only gets a direction (cosine) check, not a magnitude
    # (rel_l2) one: Keras' AdamW places epsilon differently than PyTorch's, so the two
    # frameworks' step-1 update *magnitudes* genuinely differ for small-gradient
    # parameters, converging to <1% difference by step ~1000 -- not a porting bug.
    assert_pass_rate(
        results["param_delta_tiny_eps"], cosine_threshold=0.95, rel_l2_tol=0.1,
        min_pass_rate=0.9,
        label="AdamW mechanics (bias correction / decoupled weight decay), epsilon isolated out",
    )
    assert_pass_rate(
        results["param_delta"], cosine_threshold=0.75, rel_l2_tol=float("inf"),
        min_pass_rate=0.9, label="AdamW parameter delta direction at the real epsilon=1e-7",
    )

    # --- 5. BatchNorm running-stat update ---
    assert_pass_rate(
        results["bn_stats"], cosine_threshold=0.99, rel_l2_tol=5e-2,
        min_pass_rate=0.95, label="BatchNorm running-stat update",
    )
