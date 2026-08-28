"""Shared cross-framework single-training-step runner: PyTorch vs. Keras.

Used by both the toy-scale test (tests/test_tf_gradient_equivalence.py) and the
real-checkpoint/GPU test (tests/test_tf_gradient_equivalence_real.py) so the two
share one implementation of "run one AdamW(clipnorm=...) step in each framework
on the same batch and identical weights, and compare loss/gradients/optimizer
deltas/BatchNorm stats" at any model scale.
"""

from __future__ import annotations

import numpy as np

from tests.gradient_comparison_utils import compare_all


def keras_tensors_to_pt(
    tensors: dict, num_blocks: int, shortcut_nums_desc: list, species: tuple = ("human",)
) -> dict:
    """Map a {keras_var_name: array} dict to {pt_param_name: array}, mirroring
    convert_tf_weights' weight transforms (src/promoterai_torch/utils.py) so the same
    mapping works on values (already checked against convert_tf_weights in
    test_convert.py) or gradients/deltas (checked here) -- a gradient/delta of a
    reshaped/transposed tensor is just the same reshape/transpose applied to it.
    """
    out = {}
    if "dense/kernel" in tensors:
        out["stem.weight"] = tensors["dense/kernel"].T[:, :, None]
    if "dense/bias" in tensors:
        out["stem.bias"] = tensors["dense/bias"]
    for i in range(num_blocks):
        kp = "meta_former_block" if i == 0 else f"meta_former_block_{i}"
        bp = f"blocks.{i}"
        direct_pairs = [
            (f"{kp}/batch_normalization/gamma", f"{bp}.bn1.weight"),
            (f"{kp}/batch_normalization/beta", f"{bp}.bn1.bias"),
            (f"{kp}/batch_normalization/moving_mean", f"{bp}.bn1.running_mean"),
            (f"{kp}/batch_normalization/moving_variance", f"{bp}.bn1.running_var"),
            (f"{kp}/depthwise_conv1d/bias", f"{bp}.dw_conv.bias"),
            (f"{kp}/batch_normalization_1/gamma", f"{bp}.bn2.weight"),
            (f"{kp}/batch_normalization_1/beta", f"{bp}.bn2.bias"),
            (f"{kp}/batch_normalization_1/moving_mean", f"{bp}.bn2.running_mean"),
            (f"{kp}/batch_normalization_1/moving_variance", f"{bp}.bn2.running_var"),
            (f"{kp}/dense/bias", f"{bp}.ffn1.bias"),
            (f"{kp}/dense_1/bias", f"{bp}.ffn2.bias"),
        ]
        for keras_name, pt_name in direct_pairs:
            if keras_name in tensors:
                out[pt_name] = tensors[keras_name]
        if f"{kp}/depthwise_conv1d/depthwise_kernel" in tensors:
            kernel = tensors[f"{kp}/depthwise_conv1d/depthwise_kernel"]
            out[f"{bp}.dw_conv.weight"] = kernel.transpose(1, 2, 0)
        if f"{kp}/dense/kernel" in tensors:
            out[f"{bp}.ffn1.weight"] = tensors[f"{kp}/dense/kernel"].T
        if f"{kp}/dense_1/kernel" in tensors:
            out[f"{bp}.ffn2.weight"] = tensors[f"{kp}/dense_1/kernel"].T
    for j, sp in enumerate(species):
        for p_idx, n in enumerate(shortcut_nums_desc):
            pfx = f"shortcut_{sp}{n}" if sp else f"shortcut{n}"
            if f"{pfx}/kernel" in tensors:
                out[f"output_heads.{j}.projections.{p_idx}.weight"] = tensors[f"{pfx}/kernel"].T
            if f"{pfx}/bias" in tensors:
                out[f"output_heads.{j}.projections.{p_idx}.bias"] = tensors[f"{pfx}/bias"]
    return out


def run_single_step(
    keras_model,
    pt_model,
    x_np: np.ndarray,
    y_np: np.ndarray,
    *,
    num_blocks: int,
    shortcut_nums_desc: list,
    lr: float,
    wd: float,
    eps: float,
    clip_norm: float,
    device: str = "cpu",
    species: tuple = ("human",),
    tiny_eps: float = 1e-10,
) -> dict:
    """Run one AdamW(clipnorm=...) training step in Keras and in PyTorch on the same
    batch through identical (converted) weights, and return cosine/rel-L2 comparison
    results (see gradient_comparison_utils) for loss, raw/clipped gradients, BatchNorm
    running-stat updates, and AdamW parameter deltas -- both at the real epsilon (which
    Keras' AdamW places differently than PyTorch's, see notes/implementation.md) and
    with epsilon -> 0 on both sides, which isolates and verifies the AdamW mechanics
    (bias correction, decoupled weight decay) agree regardless of that quirk.
    """
    import tensorflow as tf
    import tf_keras as keras
    import torch

    from promoterai_torch.utils import clip_grad_norm_per_parameter

    def mapped(tensors):
        return keras_tensors_to_pt(tensors, num_blocks, shortcut_nums_desc, species)

    # --- Keras: one manual train step (mirrors what model.fit does internally) ---
    keras_optimizer = keras.optimizers.AdamW(
        learning_rate=lr, weight_decay=wd, epsilon=eps, clipnorm=clip_norm
    )
    x_tf, y_tf = tf.constant(x_np), tf.constant(y_np)
    with tf.GradientTape() as tape:
        pred = keras_model(x_tf, training=True)  # updates BN running stats as a side effect
        keras_loss = keras.losses.MeanSquaredError()(y_tf, pred)
    keras_vars = keras_model.trainable_variables
    keras_names = [v.name.rstrip(":0") for v in keras_vars]
    raw_grads = tape.gradient(keras_loss, keras_vars)
    clipped_grads = [tf.clip_by_norm(g, clip_norm) for g in raw_grads]
    params_before = {n: v.numpy().copy() for n, v in zip(keras_names, keras_vars)}
    keras_optimizer.apply_gradients(zip(clipped_grads, keras_vars))
    params_after = {n: v.numpy().copy() for n, v in zip(keras_names, keras_vars)}
    bn_stats_keras = {
        v.name.rstrip(":0"): v.numpy()
        for v in keras_model.non_trainable_variables
        if "moving_" in v.name
    }
    raw_grads_keras = {n: g.numpy() for n, g in zip(keras_names, raw_grads)}
    clipped_grads_keras = {n: g.numpy() for n, g in zip(keras_names, clipped_grads)}
    param_deltas_keras = {n: params_after[n] - params_before[n] for n in params_before}

    # AdamW mechanics isolation: rerun the same clipped gradients with epsilon -> 0,
    # where Keras' and PyTorch's AdamW formulas are mathematically identical.
    keras_tiny_vars = [tf.Variable(params_before[n].copy()) for n in keras_names]
    keras.optimizers.AdamW(
        learning_rate=lr, weight_decay=wd, epsilon=tiny_eps, clipnorm=clip_norm
    ).apply_gradients(zip(clipped_grads, keras_tiny_vars))
    deltas_keras_tiny = {
        n: v.numpy() - params_before[n] for n, v in zip(keras_names, keras_tiny_vars)
    }

    # --- PyTorch: one manual train step, same batch, same (converted) weights ---
    torch_device = torch.device(device)
    pt_model.to(torch_device)
    pt_model.train()
    optimizer = torch.optim.AdamW(pt_model.parameters(), lr=lr, weight_decay=wd, eps=eps)
    x_pt = torch.from_numpy(x_np).to(torch_device)
    y_pt = torch.from_numpy(y_np).to(torch_device)
    pred_pt = pt_model(x_pt)[0]  # updates BN running stats as a side effect
    loss_pt = torch.nn.functional.mse_loss(pred_pt, y_pt)
    optimizer.zero_grad()
    loss_pt.backward()
    raw_grads_pt = {n: p.grad.detach().clone() for n, p in pt_model.named_parameters()}
    params_before_pt = {n: p.detach().clone() for n, p in pt_model.named_parameters()}
    clip_grad_norm_per_parameter(pt_model.parameters(), max_norm=clip_norm)
    clipped_grads_pt = {n: p.grad.detach().clone() for n, p in pt_model.named_parameters()}
    optimizer.step()
    param_deltas_pt = {
        n: (p.detach() - params_before_pt[n]).cpu().numpy()
        for n, p in pt_model.named_parameters()
    }
    bn_stats_pt = {}
    for i, block in enumerate(pt_model.blocks):
        bn_stats_pt[f"blocks.{i}.bn1.running_mean"] = block.bn1.running_mean.cpu().numpy()
        bn_stats_pt[f"blocks.{i}.bn1.running_var"] = block.bn1.running_var.cpu().numpy()
        bn_stats_pt[f"blocks.{i}.bn2.running_mean"] = block.bn2.running_mean.cpu().numpy()
        bn_stats_pt[f"blocks.{i}.bn2.running_var"] = block.bn2.running_var.cpu().numpy()

    # AdamW mechanics isolation on the PyTorch side: fresh leaves at the pre-step
    # values, same clipped gradients, epsilon -> 0.
    deltas_pt_tiny = {}
    for name, grad in clipped_grads_pt.items():
        p = torch.nn.Parameter(params_before_pt[name].clone())
        p.grad = grad.clone()
        torch.optim.AdamW([p], lr=lr, weight_decay=wd, eps=tiny_eps).step()
        deltas_pt_tiny[name] = (p.detach() - params_before_pt[name]).cpu().numpy()

    raw_grads_pt_np = {n: g.cpu().numpy() for n, g in raw_grads_pt.items()}
    clipped_grads_pt_np = {n: g.cpu().numpy() for n, g in clipped_grads_pt.items()}

    return {
        "loss_pt": float(loss_pt.item()),
        "loss_keras": float(keras_loss.numpy()),
        "raw_grad": compare_all(raw_grads_pt_np, mapped(raw_grads_keras)),
        "clipped_grad": compare_all(clipped_grads_pt_np, mapped(clipped_grads_keras)),
        "param_delta": compare_all(param_deltas_pt, mapped(param_deltas_keras)),
        "param_delta_tiny_eps": compare_all(deltas_pt_tiny, mapped(deltas_keras_tiny)),
        "bn_stats": compare_all(bn_stats_pt, mapped(bn_stats_keras)),
    }
