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

For a multi-species checkpoint (e.g. human + mouse), real training draws one
species' batch at a time -- weighted by dataset size, never every species
summed every step. Illumina's tfrecords.py handles this with a *soft* zero:
the inactive species' loss term is still computed (against a dummy target,
broadcasting to zero) and multiplied by sample_weight=0, which zeroes its loss
value and gradient but keeps it in the graph, so its output head still gets a
real, zero gradient every step -- meaning Keras' AdamW still applies weight
decay to it, batch after batch, regardless of which species happens to be
active. train.py's compute_loss mirrors this exactly (see its docstring); a
hard skip would instead leave that head with grad=None, and PyTorch's
optimizer would then skip weight decay for it entirely on an inactive batch --
a real behavioral divergence from Illumina's training this guards against.

This test mirrors that exactly: it runs one full comparison per output head,
with that head active (weight 1.0) and every other head weighted 0.0 (target
replaced with a dummy placeholder purely to keep shapes broadcastable),
reloading fresh model weights each time so no iteration's optimizer step leaks
into the next. Inactive heads' projection parameters are therefore expected to
end each iteration with at most a weight-decay-only change from AdamW's
gradient-independent `variable -= variable * weight_decay * lr` term (at this
lr/weight_decay, that's ~2.5e-9 relative to the variable -- often literally
unrepresentable in float32 against a ~0.01-0.1 weight, so it may round back
to identically zero on both sides) -- either way, the two frameworks must
still agree, unlike a hard skip, which would guarantee exactly zero on both
sides for a different (wrong) reason. That agreement is what's under test,
not any particular nonzero magnitude.

Thresholds are looser than tests/test_tf_gradient_equivalence.py's: gradients
accumulate more floating-point divergence through 24 stacked blocks than
through 8, per alphagenome-pytorch's tests/README.md tolerance notes.
"""

import numpy as np
import pytest

from tests.gradient_comparison_utils import assert_pass_rate, report_top_offenders
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
    """One AdamW(clipnorm=...) step per output head, real checkpoint scale."""
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
    _, args = load_pretrained(out_pt)
    num_blocks = args["num_blocks"]
    shortcut_layer_freq = args.get("shortcut_layer_freq", 4)
    shortcut_nums_desc = list(range(num_blocks, 0, -shortcut_layer_freq))
    output_dims = args["output_dims"]
    # Ground truth from convert_tf_weights: the exact order output_heads was built
    # in, which need not be human/hg38-first -- see normalize_keras_outputs.
    species_order = tuple(args["species_order"])

    rng = np.random.default_rng(0)
    idx = rng.integers(0, 4, size=(gradient_batch_size, gradient_input_length))
    x_np = np.eye(4, dtype="float32")[idx]  # (B, L, 4)

    lr, wd, eps, clip_norm = 5e-4, 5e-6, 1e-7, 1e-4

    for active_idx in range(len(output_dims)):
        # Fresh model instances per head so each iteration starts from the same
        # pristine checkpoint weights, not the previous iteration's post-step state.
        pt_model, _ = load_pretrained(out_pt)
        keras_model = keras.models.load_model(keras_savedmodel_path)

        y_np = [
            rng.normal(
                size=(gradient_batch_size, gradient_output_length, output_dim)
            ).astype("float32")
            if j == active_idx
            else np.zeros((gradient_batch_size, 1, 1), dtype="float32")
            for j, output_dim in enumerate(output_dims)
        ]
        weights = [1.0 if j == active_idx else 0.0 for j in range(len(output_dims))]

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
            weights=weights,
            species_order=species_order,
        )

        suffix = f" (head {active_idx}/{len(output_dims)} active)"

        # Printed unconditionally (run with -s) so a forward-loss mismatch below is
        # immediately localized to "predictions genuinely diverge at this scale" vs.
        # "something upstream of the loss is wrong" -- see report_top_offenders.
        print(f"\n[diagnostic] per-head prediction agreement{suffix}:")
        print(report_top_offenders(results["prediction"], k=len(results["prediction"])))

        # --- 1. Forward loss ---
        np.testing.assert_allclose(
            results["loss_pt"], results["loss_keras"], atol=1e-3, rtol=1e-3,
            err_msg=f"forward-pass MSE loss differs between PyTorch and Keras{suffix}",
        )

        # --- 2/3. Raw and post-clip gradients ---
        assert_pass_rate(
            results["raw_grad"], cosine_threshold=0.99, rel_l2_tol=5e-2,
            min_pass_rate=0.95, label=f"raw gradients{suffix}",
        )
        assert_pass_rate(
            results["clipped_grad"], cosine_threshold=0.99, rel_l2_tol=5e-2,
            min_pass_rate=0.95, label=f"post-clip gradients{suffix}",
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
            label=(
                "AdamW mechanics (bias correction / decoupled weight decay), "
                f"epsilon isolated out{suffix}"
            ),
        )
        assert_pass_rate(
            results["param_delta"], cosine_threshold=0.75, rel_l2_tol=float("inf"),
            min_pass_rate=0.9,
            label=f"AdamW parameter delta direction at the real epsilon=1e-7{suffix}",
        )

        # --- 5. BatchNorm running-stat update ---
        assert_pass_rate(
            results["bn_stats"], cosine_threshold=0.99, rel_l2_tol=5e-2,
            min_pass_rate=0.95, label=f"BatchNorm running-stat update{suffix}",
        )
