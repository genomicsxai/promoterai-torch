"""
Cross-framework single-step training equivalence: PyTorch vs. Keras (toy scale).

Per reviewer finding (Al-Murphy, PR #120 on the blog post), this runs one
optimizer step in train mode on a fixed batch through numerically identical
(converted) weights in both frameworks and compares loss, raw gradients,
post-clip gradients, resulting parameter deltas, and BatchNorm running-stat
updates. It exercises exactly the two behaviors fixed in fd0f2e3 (per-parameter
gradient clipping vs. global clipnorm; BatchNorm momentum 0.01 vs. 0.1) plus
the AdamW epsilon fix in 18c009a, using real cross-framework execution rather
than the PyTorch-only tests added alongside those fixes.

Uses a small (8-block, dim-16) model on random weights/data purely to check
gradient/optimizer *mechanics* cheaply and deterministically -- it does not
exercise anything that only shows up at the published scale (24 blocks,
dim 1024) or on real genomic data. tests/test_tf_gradient_equivalence_real.py
runs the same comparison against a real converted checkpoint on a GPU.

Comparisons use cosine similarity (direction) and relative L2 norm (magnitude)
per tensor, aggregated with a required pass *rate* across tensors -- see
tests/gradient_comparison_utils.py -- rather than elementwise
np.testing.assert_allclose. That's because both stem paths fuse a ReLU right
after their linear op, so on an unlucky weight draw a handful of pre-activation
values can land close enough to zero that ordinary TF-vs-PyTorch summation-order
floating-point noise flips which side of the ReLU boundary they're on in one
framework but not the other, killing or admitting gradient for that one element
only -- expected, harmless framework noise, not a porting bug, that a strict
per-element assert would fail on.

Skipped automatically when tf-keras is not installed.
"""

import numpy as np
import pytest

from tests.gradient_comparison_utils import assert_pass_rate
from tests.keras_pytorch_step import run_single_step
from tests.test_convert import _build_tf_keras_model, _save_savedmodel

pytest.importorskip("tf_keras", reason="tf-keras not installed")


def test_single_step_gradient_and_optimizer_equivalence(tmp_path):
    """One AdamW(clipnorm=...) step on identical weights/batch: PyTorch vs. Keras
    must agree on loss, raw grads, clipped grads, param deltas, and BN stats.
    """
    import tensorflow as tf

    from promoterai_torch.utils import convert_tf_weights, load_pretrained

    # Seed Keras' weight initialization. Without this, _build_tf_keras_model draws
    # from TF's global RNG, whose state depends on how many prior TF random ops ran
    # earlier in the process -- identical when this file runs standalone (always the
    # "first" draw), but different when it runs after other TF-touching tests in the
    # full suite. Seeding makes the draw -- and thus which elements happen to sit near
    # a ReLU boundary -- reproducible; the pass-rate/cosine checks below (rather than a
    # strict per-element assert) are the actual fix for the boundary case itself.
    tf.keras.utils.set_random_seed(0)

    num_blocks, model_dim, output_dim = 8, 16, 4
    shortcut_layer_freq = 4
    input_len, output_len = 64, 56
    output_crop = input_len - output_len
    batch_size = 4
    lr, wd, eps, clip_norm = 5e-4, 5e-6, 1e-7, 1e-4
    shortcut_nums_desc = list(range(num_blocks, 0, -shortcut_layer_freq))

    keras_model = _build_tf_keras_model(
        num_blocks=num_blocks,
        model_dim=model_dim,
        output_dims=(output_dim,),
        output_crop=output_crop,
        shortcut_layer_freq=shortcut_layer_freq,
    )
    _save_savedmodel(keras_model, str(tmp_path / "keras_model"))
    out_pt = str(tmp_path / "model.pt")
    convert_tf_weights(
        str(tmp_path / "keras_model"),
        out_pt,
        input_length=input_len,
        output_length=output_len,
    )
    pt_model, _ = load_pretrained(out_pt)

    rng = np.random.default_rng(0)
    idx = rng.integers(0, 4, size=(batch_size, input_len))
    x_np = np.eye(4, dtype="float32")[idx]  # (B, L, 4)
    y_np = [rng.normal(size=(batch_size, output_len, output_dim)).astype("float32")]

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
    )

    # --- 1. Forward loss ---
    np.testing.assert_allclose(
        results["loss_pt"], results["loss_keras"], atol=1e-4, rtol=1e-4,
        err_msg="forward-pass MSE loss differs between PyTorch and Keras",
    )

    # --- 2. Raw (pre-clip) gradients ---
    assert_pass_rate(
        results["raw_grad"], cosine_threshold=0.999, rel_l2_tol=1e-2,
        min_pass_rate=1.0, label="raw gradients",
    )

    # --- 3. Post-clip gradients (validates clip_grad_norm_per_parameter vs. Keras clipnorm) ---
    assert_pass_rate(
        results["clipped_grad"], cosine_threshold=0.999, rel_l2_tol=1e-2,
        min_pass_rate=1.0, label="post-clip gradients",
    )

    # --- 4. AdamW parameter deltas ---
    #
    # keras.optimizers.AdamW.update_step (tf_keras/src/optimizers/adamw.py) hardcodes
    #   alpha_t = lr * sqrt(1 - beta2**t) / (1 - beta1**t)
    #   update  = alpha_t * m / (sqrt(v) + epsilon)          # m, v are the *biased* moments
    # which only equals the textbook bias-corrected Adam update that torch.optim.AdamW
    # computes (update = lr * m_hat / (sqrt(v_hat) + epsilon)) if epsilon is rescaled by
    # sqrt(1 - beta2**t) — Keras' AdamW has no such option (unlike keras.optimizers.Adam's
    # adaptive_epsilon), so with a shared epsilon=1e-7 the two frameworks' step-1 updates
    # differ by ~1/sqrt(1 - 0.999**1) ≈ 32x in how much epsilon damps the update, shrinking
    # to <1% by step ~1000 as beta2**t → 0 (see notes/implementation.md). This is a
    # framework-inherent AdamW quirk, not a porting bug — confirmed by "param_delta_tiny_eps"
    # below, which reruns the same clipped gradients through fresh step-1 optimizers with
    # epsilon -> 0 (where the two formulas coincide exactly), isolating and verifying the
    # shared bias-correction/decoupled-weight-decay mechanics.
    assert_pass_rate(
        results["param_delta_tiny_eps"], cosine_threshold=0.99, rel_l2_tol=5e-2,
        min_pass_rate=0.95,
        label="AdamW mechanics (bias correction / decoupled weight decay), epsilon isolated out",
    )

    # With the real epsilon=1e-7 (matching build_train_optimizer), the quirk above makes
    # the *magnitude* of small-gradient parameters' step-1 updates genuinely differ (rel_l2
    # regularly exceeds 100%), so rel_l2 is not asserted here. But epsilon only *damps* the
    # update -- it doesn't change the sign of m -- so the update's *direction* should still
    # agree even though the fixes themselves already gave a hard forward/gradient guarantee
    # above; cosine similarity alone is the meaningful, real check for that.
    assert_pass_rate(
        results["param_delta"], cosine_threshold=0.75, rel_l2_tol=float("inf"),
        min_pass_rate=0.95, label="AdamW parameter delta direction at the real epsilon=1e-7",
    )
    rel_l2s = [r.rel_l2 for r in results["param_delta"]]
    print(
        f"\n[diagnostic] step-1 AdamW delta relative-L2 magnitude vs. Keras at the real "
        f"epsilon=1e-7 (median={np.median(rel_l2s):.1%}) — expected to be large due to the "
        "epsilon-placement quirk documented above; see the param_delta_tiny_eps check above "
        "for proof the underlying AdamW mechanics agree once epsilon is isolated out."
    )

    # --- 5. BatchNorm running-stat update (validates momentum=0.01 <-> Keras momentum=0.99) ---
    assert_pass_rate(
        results["bn_stats"], cosine_threshold=0.999, rel_l2_tol=1e-2,
        min_pass_rate=1.0, label="BatchNorm running-stat update",
    )
