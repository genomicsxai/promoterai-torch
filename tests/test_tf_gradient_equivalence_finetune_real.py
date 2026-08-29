"""
Cross-framework single-step FINETUNING equivalence on a REAL checkpoint (GPU).

tests/test_tf_gradient_equivalence_real.py assumes a from-scratch, fully-trainable
model (train.py's scenario): every parameter trainable, every BatchNorm layer
computing live batch statistics in train mode, symmetrically on both frameworks.
An actually fine-tuned checkpoint (e.g. hg38_finetune, hg38_mm10_finetune) doesn't
match that: most of the model -- the backbone and every non-primary species head --
is frozen (non-trainable) in the SavedModel itself. Keras forces a non-trainable
BatchNormalization layer into inference mode (fixed running stats) regardless of
the training=True argument passed to the model call, while a blanket
pt_model.train() puts every PyTorch BatchNorm layer into live-batch-statistics
mode with no such exception -- a genuine structural mismatch (not a rounding-order
artifact), which is what test_tf_gradient_equivalence_real.py's forward-pass
comparison surfaces as a large, deterministic (immune to TF32/oneDNN toggling)
divergence when pointed at a fine-tuned checkpoint. See the finetune section in
notes/implementation.md for the full diagnosis.

This test instead mirrors finetune.py/TwinModel exactly: one AdamW(clipnorm=1.0)
step on a ref/alt diff through output_heads[0] only, with the backbone and every
other species head frozen on both sides (TwinModel.train()'s
base_model.eval() / output_heads[0].train() split on the PyTorch side; Keras'
own non-trainable-BatchNorm-forces-inference-mode behavior on the Keras side --
no special-casing needed there).

Requires:
- a real Illumina PromoterAI Keras SavedModel that was itself fine-tuned (most
  variables non-trainable), licensed and not distributed with this repo, passed
  via --keras-savedmodel-path
- enough memory to hold and backprop through the full model -- practically a
  GPU, hence the separate --device/--gradient-batch-size options

Skipped automatically unless --keras-savedmodel-path is given (including in CI,
where the licensed model is never available).

    uv sync --group dev --extra convert
    uv run pytest tests/test_tf_gradient_equivalence_finetune_real.py -v -s \\
        --keras-savedmodel-path /path/to/promoterai_keras_finetune_model \\
        --device cuda --gradient-batch-size 2

Inputs are random one-hot sequences and random z-score targets, not real
variants -- this checks optimizer/gradient *mechanics* at real scale, not model
predictions (those are validated separately in examples/, using real variants
and the model's real predictions).
"""

import numpy as np
import pytest

from tests.gradient_comparison_utils import assert_pass_rate
from tests.keras_pytorch_step import force_tf_cpu, run_single_finetune_step

pytest.importorskip("tf_keras", reason="tf-keras not installed")


def test_real_finetune_checkpoint_single_step_gradient_equivalence(
    tmp_path,
    keras_savedmodel_path,
    gradient_device,
    gradient_batch_size,
    gradient_input_length,
    gradient_output_length,
):
    """One AdamW(clipnorm=1.0) TwinModel finetuning step, real fine-tuned checkpoint."""
    force_tf_cpu()  # before any TF op, including inside convert_tf_weights below
    import tf_keras as keras

    from promoterai_torch.architecture import TwinModel
    from promoterai_torch.utils import convert_tf_weights, load_pretrained

    out_pt = str(tmp_path / "model.pt")
    # input_length AND output_length are both required here -- convert_tf_weights only
    # derives output_crop (needed to produce the right-length prediction to diff/mean
    # over) when both are given; omitting output_length would silently leave the
    # PyTorch model's output uncropped while Keras' own saved output_length stays
    # correct, producing a shape-driven mismatch that looks like a real bug but isn't.
    convert_tf_weights(
        keras_savedmodel_path,
        out_pt,
        input_length=gradient_input_length,
        output_length=gradient_output_length,
    )
    _, args = load_pretrained(out_pt)
    num_blocks = args["num_blocks"]
    shortcut_layer_freq = args.get("shortcut_layer_freq", 4)
    shortcut_nums_desc = list(range(num_blocks, 0, -shortcut_layer_freq))
    species_order = tuple(args["species_order"])

    pt_model, _ = load_pretrained(out_pt)
    twin_model = TwinModel(pt_model)
    keras_model = keras.models.load_model(keras_savedmodel_path)

    rng = np.random.default_rng(0)
    idx_ref = rng.integers(0, 4, size=(gradient_batch_size, gradient_input_length))
    idx_alt = rng.integers(0, 4, size=(gradient_batch_size, gradient_input_length))
    x_ref_np = np.eye(4, dtype="float32")[idx_ref]
    x_alt_np = np.eye(4, dtype="float32")[idx_alt]
    y_np = rng.normal(size=(gradient_batch_size,)).astype("float32")

    lr, wd, eps, clip_norm = 5e-4, 5e-6, 1e-7, 1.0  # matches build_finetune_optimizer/_run_epoch

    results = run_single_finetune_step(
        keras_model,
        twin_model,
        x_ref_np,
        x_alt_np,
        y_np,
        num_blocks=num_blocks,
        shortcut_nums_desc=shortcut_nums_desc,
        lr=lr,
        wd=wd,
        eps=eps,
        clip_norm=clip_norm,
        device=gradient_device,
        species_order=species_order,
    )

    # --- 1. Forward loss ---
    np.testing.assert_allclose(
        results["loss_pt"], results["loss_keras"], atol=1e-3, rtol=1e-3,
        err_msg="forward-pass MSE loss differs between PyTorch and Keras",
    )

    # --- 2/3. Raw and post-clip gradients (output_heads[0] only -- everything else
    # is frozen and correctly excluded on both sides) ---
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
    # See notes/implementation.md's AdamW epsilon section: the real epsilon=1e-7 case
    # only gets a direction (cosine) check, not a magnitude one, for the same reason
    # as the from-scratch test -- Keras' AdamW places epsilon differently than
    # PyTorch's, so step-1 update *magnitudes* genuinely differ for small-gradient
    # parameters, converging to <1% difference by step ~1000 -- not a porting bug.
    assert_pass_rate(
        results["param_delta_tiny_eps"], cosine_threshold=0.95, rel_l2_tol=0.1,
        min_pass_rate=0.9,
        label="AdamW mechanics (bias correction / decoupled weight decay), epsilon isolated out",
    )
    assert_pass_rate(
        results["param_delta"], cosine_threshold=0.75, rel_l2_tol=float("inf"),
        min_pass_rate=0.9,
        label="AdamW parameter delta direction at the real epsilon=1e-7",
    )

    # --- 5. Frozen backbone must not move at all, on either side ---
    assert_pass_rate(
        results["bn_unchanged_keras"], cosine_threshold=1.0, rel_l2_tol=1e-9,
        min_pass_rate=1.0,
        label="Keras backbone BatchNorm running stats changed despite being frozen",
    )
    assert_pass_rate(
        results["bn_unchanged_pt"], cosine_threshold=1.0, rel_l2_tol=1e-9,
        min_pass_rate=1.0,
        label="PyTorch backbone BatchNorm running stats changed despite TwinModel.eval()",
    )
