"""
Cross-framework single-step FINETUNING equivalence: PyTorch vs. Keras (toy scale).

Toy-scale counterpart to tests/test_tf_gradient_equivalence_finetune_real.py --
proves run_single_finetune_step's mechanics (TwinModel's frozen backbone /
single-trainable-head split, matched against Keras' own non-trainable-layers
forced into inference mode) are right at a scale that runs in seconds on a
laptop, using an explicitly frozen toy model to simulate a fine-tuned
checkpoint. It doesn't exercise anything that only shows up at full depth or
against a real Illumina checkpoint.

Skipped automatically when tf-keras is not installed.
"""

import numpy as np
import pytest

from tests.gradient_comparison_utils import assert_exact_match, assert_pass_rate
from tests.keras_pytorch_step import run_single_finetune_step
from tests.test_convert import _build_tf_keras_model, _save_savedmodel

pytest.importorskip("tf_keras", reason="tf-keras not installed")


def test_single_finetune_step_gradient_and_optimizer_equivalence(tmp_path):
    """One AdamW(clipnorm=1.0) TwinModel-style step: PyTorch vs. Keras must agree,
    with the backbone frozen (matching an actual fine-tuned checkpoint) on both sides.
    """
    import tensorflow as tf

    from promoterai_torch.architecture import TwinModel
    from promoterai_torch.utils import convert_tf_weights, load_pretrained

    tf.keras.utils.set_random_seed(0)

    num_blocks, model_dim, output_dim = 8, 16, 4
    shortcut_layer_freq = 4
    input_len = 64
    batch_size = 4
    lr, wd, eps, clip_norm = 5e-4, 5e-6, 1e-7, 1.0
    shortcut_nums_desc = list(range(num_blocks, 0, -shortcut_layer_freq))

    keras_model = _build_tf_keras_model(
        num_blocks=num_blocks,
        model_dim=model_dim,
        output_dims=(output_dim,),
        shortcut_layer_freq=shortcut_layer_freq,
    )
    # Simulate a fine-tuned checkpoint: freeze the stem and every backbone block,
    # leaving only the output head's own shortcut projections trainable -- matching
    # TwinModel's own freeze (everything except output_heads[0]).
    for layer in keras_model.layers:
        if layer.name == "dense" or layer.name.startswith("meta_former_block"):
            layer.trainable = False

    _save_savedmodel(keras_model, str(tmp_path / "keras_model"))
    out_pt = str(tmp_path / "model.pt")
    convert_tf_weights(
        str(tmp_path / "keras_model"), out_pt, input_length=input_len, output_length=input_len
    )
    pt_model, _ = load_pretrained(out_pt)
    twin_model = TwinModel(pt_model)

    rng = np.random.default_rng(0)
    idx_ref = rng.integers(0, 4, size=(batch_size, input_len))
    idx_alt = rng.integers(0, 4, size=(batch_size, input_len))
    x_ref_np = np.eye(4, dtype="float32")[idx_ref]
    x_alt_np = np.eye(4, dtype="float32")[idx_alt]
    y_np = rng.normal(size=(batch_size,)).astype("float32")

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
        species_order=("human",),
    )

    np.testing.assert_allclose(
        results["loss_pt"], results["loss_keras"], atol=1e-4, rtol=1e-4,
        err_msg="forward-pass MSE loss differs between PyTorch and Keras",
    )

    assert_pass_rate(
        results["raw_grad"], cosine_threshold=0.999, rel_l2_tol=1e-2,
        min_pass_rate=1.0, label="raw gradients",
    )
    assert_pass_rate(
        results["clipped_grad"], cosine_threshold=0.999, rel_l2_tol=1e-2,
        min_pass_rate=1.0, label="post-clip gradients",
    )
    assert_pass_rate(
        results["param_delta_tiny_eps"], cosine_threshold=0.99, rel_l2_tol=5e-2,
        min_pass_rate=0.95,
        label="AdamW mechanics (bias correction / decoupled weight decay), epsilon isolated out",
    )
    assert_pass_rate(
        results["param_delta"], cosine_threshold=0.75, rel_l2_tol=float("inf"),
        min_pass_rate=0.95, label="AdamW parameter delta direction at the real epsilon=1e-7",
    )

    # The frozen backbone must not move at all, on either side.
    assert_exact_match(
        results["bn_unchanged_keras"],
        label="Keras backbone BatchNorm running stats changed despite being frozen",
    )
    assert_exact_match(
        results["bn_unchanged_pt"],
        label="PyTorch backbone BatchNorm running stats changed despite TwinModel.eval()",
    )

    # Exactly the output head's 2*len(shortcut_nums_desc) tensors should be trainable.
    assert len(results["raw_grad"]) == 2 * len(shortcut_nums_desc)
