"""Shared cross-framework single-training-step runner: PyTorch vs. Keras.

run_single_step is used by both the toy-scale test
(tests/test_tf_gradient_equivalence.py) and the real-checkpoint/GPU test
(tests/test_tf_gradient_equivalence_real.py) so the two share one
implementation of "run one AdamW(clipnorm=...) step in each framework on the
same batch and identical weights, and compare loss/gradients/optimizer
deltas/BatchNorm stats" at any model scale -- this assumes a from-scratch,
fully-trainable model, matching train.py.

run_single_finetune_step is the equivalent for finetune.py's TwinModel
scenario (backbone and every non-primary species head frozen, only
output_heads[0] trainable), used by
tests/test_tf_gradient_equivalence_finetune_real.py. The two aren't
interchangeable: running a fine-tuned checkpoint (most of the model frozen)
through run_single_step's blanket-trainable assumption produces a large,
spurious mismatch -- see the finetune section in notes/implementation.md.
"""

from __future__ import annotations

import numpy as np

from tests.gradient_comparison_utils import compare_all


def force_tf_cpu() -> None:
    """Make TensorFlow ignore any visible GPUs for the rest of the process.

    Must be called before TF touches a GPU device at all (including inside
    convert_tf_weights/keras.models.load_model, not just this module's own ops) --
    call it first thing, before any other TF-touching code runs.

    Two independent reasons this matters, not just "avoid contending with PyTorch
    for the same GPU's VRAM": tf_keras.layers.DepthwiseConv1D implements dilated
    convolution via SpaceToBatchND, which multiplies the *batch dimension* by the
    dilation rate for that op -- up to 512-1024x for this model's deepest blocks
    -- so a GPU that comfortably holds the equivalent PyTorch computation can
    still OOM on the Keras side alone. CPU/system RAM has none of this problem in
    practice (much more headroom, and no such GPU-specific kernel blowup).
    """
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")


def normalize_keras_outputs(
    pred, species_order: tuple | None = None
) -> tuple[list, tuple | None]:
    """Return (every species head's output tensor, that head's species name if known),
    ordered to line up positionally with the PyTorch side's output_heads.

    Multi-species Keras functional models (e.g. human+mouse) return a dict keyed by
    species name rather than a plain tensor/tuple. The PyTorch side's tuple output has
    no such names -- its head order is convert_tf_weights' species_order, i.e. the
    *first-appearance order of shortcut_{species}{N} weights in the SavedModel's own
    weight list*, which is not necessarily human/hg38-first (that's only a convention
    AGENTS.md documents for authoring new checkpoints, not something convert_tf_weights
    enforces). Passing that real species_order in (from load_pretrained's checkpoint
    args) is therefore the only reliable way to line the two sides' heads up -- a
    human/mouse name heuristic on the dict's own keys can silently pick a different
    order than what convert_tf_weights actually used, aligning the wrong head's
    prediction/target pair without any shape mismatch to catch it.

    Falls back to the human/hg38-first heuristic only when no species_order is given,
    or it doesn't match the dict's actual key set (e.g. a checkpoint converted before
    species_order was recorded).
    """
    if isinstance(pred, dict):
        if species_order is not None and set(map(str, pred)) == set(
            map(str, species_order)
        ):
            key_by_str = {str(k): k for k in pred}
            ordered_keys = [key_by_str[str(s)] for s in species_order]
        else:
            def _rank(key):
                key_str = str(key).lower()
                if "human" in key_str or "hg38" in key_str:
                    return (0, key_str)
                if "mouse" in key_str or "mm10" in key_str:
                    return (1, key_str)
                return (2, key_str)

            ordered_keys = sorted(pred, key=_rank)
        return [pred[k] for k in ordered_keys], tuple(str(k) for k in ordered_keys)
    if isinstance(pred, (list, tuple)):
        return list(pred), None
    return [pred], None


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
    y_np: list,
    *,
    num_blocks: int,
    shortcut_nums_desc: list,
    lr: float,
    wd: float,
    eps: float,
    clip_norm: float,
    device: str = "cpu",
    species: tuple = ("human",),
    species_order: tuple | None = None,
    tiny_eps: float = 1e-10,
    weights: list | None = None,
) -> dict:
    """Run one AdamW(clipnorm=...) training step in Keras and in PyTorch on the same
    batch through identical (converted) weights, and return cosine/rel-L2 comparison
    results (see gradient_comparison_utils) for loss, raw/clipped gradients, BatchNorm
    running-stat updates, and AdamW parameter deltas -- both at the real epsilon (which
    Keras' AdamW places differently than PyTorch's, see notes/implementation.md) and
    with epsilon -> 0 on both sides, which isolates and verifies the AdamW mechanics
    (bias correction, decoupled weight decay) agree regardless of that quirk.

    y_np is one target array per output head, in the same order as pred_pt's tuple
    (and the Keras dict once ordered by species_order -- see normalize_keras_outputs).
    species_order should be the ground-truth order from load_pretrained's checkpoint
    args (convert_tf_weights' species_order), so the Keras dict's per-species outputs
    line up positionally with pred_pt's tuple and with keras_tensors_to_pt's
    shortcut_{species}{N} weight lookups -- required for any multi-species real
    checkpoint, where species_order need not be human/hg38-first. weights is
    one scalar per head (default: all 1.0), mirroring train.py's compute_loss w_tuple /
    dataset.py's sample_weight: a species not in this batch gets weight 0.0, zeroing
    both its loss contribution and (via the chain rule) its gradient, while a head
    whose target has last dim 1 is treated as that species' dummy placeholder and its
    target is substituted with zeros purely to keep the shapes broadcastable -- the
    *value* used there doesn't matter once its weight is 0. This mirrors Illumina's
    tfrecords.py exactly (soft zero, not a hard skip): a real multi-species training
    step never sums every species' loss, but every head still gets a real (if zero)
    gradient every step, so weight decay applies uniformly regardless of which species
    happens to be active that step (see compute_loss's docstring for why that matters).
    """
    force_tf_cpu()
    import tensorflow as tf
    import tf_keras as keras
    import torch

    from promoterai_torch.utils import clip_grad_norm_per_parameter

    # --- Keras: one manual train step (mirrors what model.fit does internally) ---
    keras_optimizer = keras.optimizers.AdamW(
        learning_rate=lr, weight_decay=wd, epsilon=eps, clipnorm=clip_norm
    )
    x_tf = tf.constant(x_np)
    y_tf_list = [tf.constant(y) for y in y_np]
    head_weights = weights if weights is not None else [1.0] * len(y_np)
    with tf.GradientTape() as tape:
        pred = keras_model(x_tf, training=True)  # updates BN running stats as a side effect
        pred_list, derived_species = normalize_keras_outputs(pred, species_order)
        # Mirrors train.py's compute_loss / dataset.py's _prepare_sample exactly: the
        # weight (not the dummy target) is what zeroes an inactive species' loss value
        # and, via the chain rule, its gradient -- the target substitution below is only
        # to keep shapes broadcastable, since 0 * anything is 0 regardless of what that
        # "anything" was. Crucially, the term still stays in the graph, so that head
        # gets a real, zero gradient rather than none, and weight decay still applies
        # to it this step, matching Illumina's sample_weight=0 convention rather than a
        # hard skip -- see compute_loss's docstring for why that distinction matters.
        mse = keras.losses.MeanSquaredError()
        losses = [
            w * mse(y if y.shape[-1] != 1 else tf.zeros_like(p), p)
            for w, y, p in zip(head_weights, y_tf_list, pred_list)
        ]
        keras_loss = tf.add_n(losses)

    effective_species = derived_species if derived_species is not None else (species_order or species)

    def mapped(tensors):
        return keras_tensors_to_pt(tensors, num_blocks, shortcut_nums_desc, effective_species)

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
    # force_tf_cpu() above pins Keras to full-precision float32 on CPU (TF32 is a
    # GPU-only reduced-precision matmul/conv mode, never used on CPU) -- if this ran
    # on a CUDA device with PyTorch's TF32 paths left enabled, every conv/matmul would
    # run at ~10 mantissa bits instead of float32's 23, and 24 blocks of BatchNorm
    # feeding its *live* batch statistics forward (unlike eval mode's fixed running
    # stats) would keep re-measuring and re-applying that per-layer precision loss,
    # compounding a small per-op rounding difference into a large, but still
    # direction-correlated, final scale drift -- exactly the pattern this test's
    # "prediction" diagnostic previously caught (cosine~0.92, rel_l2~47%). Disabling
    # TF32 makes the comparison apples-to-apples regardless of PyTorch/cuDNN version
    # defaults, which have changed across releases.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch_device = torch.device(device)
    pt_model.to(torch_device)
    pt_model.train()
    optimizer = torch.optim.AdamW(pt_model.parameters(), lr=lr, weight_decay=wd, eps=eps)
    x_pt = torch.from_numpy(x_np).to(torch_device)
    y_pt_list = [torch.from_numpy(y).to(torch_device) for y in y_np]
    preds_pt = pt_model(x_pt)  # tuple, one per head; updates BN stats as a side effect
    pred_names = [f"head{j}" for j in range(len(preds_pt))]
    preds_pt_np = {n: p.detach().cpu().numpy() for n, p in zip(pred_names, preds_pt)}
    preds_keras_np = {n: p.numpy() for n, p in zip(pred_names, pred_list)}
    loss_pt = sum(
        w * torch.nn.functional.mse_loss(
            p, y if y.shape[-1] != 1 else torch.zeros_like(p)
        )
        for w, y, p in zip(head_weights, y_pt_list, preds_pt)
    )
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
        "prediction": compare_all(preds_pt_np, preds_keras_np),
        "raw_grad": compare_all(raw_grads_pt_np, mapped(raw_grads_keras)),
        "clipped_grad": compare_all(clipped_grads_pt_np, mapped(clipped_grads_keras)),
        "param_delta": compare_all(param_deltas_pt, mapped(param_deltas_keras)),
        "param_delta_tiny_eps": compare_all(deltas_pt_tiny, mapped(deltas_keras_tiny)),
        "bn_stats": compare_all(bn_stats_pt, mapped(bn_stats_keras)),
    }


def _backbone_bn_stats(base_model) -> dict:
    """Snapshot every backbone BatchNorm running stat, keyed to match keras_tensors_to_pt."""
    stats = {}
    for i, block in enumerate(base_model.blocks):
        stats[f"blocks.{i}.bn1.running_mean"] = block.bn1.running_mean.detach().cpu().numpy().copy()
        stats[f"blocks.{i}.bn1.running_var"] = block.bn1.running_var.detach().cpu().numpy().copy()
        stats[f"blocks.{i}.bn2.running_mean"] = block.bn2.running_mean.detach().cpu().numpy().copy()
        stats[f"blocks.{i}.bn2.running_var"] = block.bn2.running_var.detach().cpu().numpy().copy()
    return stats


def run_single_finetune_step(
    keras_model,
    pt_twin_model,
    x_ref_np: np.ndarray,
    x_alt_np: np.ndarray,
    y_np: np.ndarray,
    *,
    num_blocks: int,
    shortcut_nums_desc: list,
    lr: float,
    wd: float,
    eps: float,
    clip_norm: float,
    device: str = "cpu",
    species_order: tuple | None = None,
    tiny_eps: float = 1e-10,
) -> dict:
    """Run one finetune.py-equivalent AdamW(clipnorm=...) step: ref/alt diff through
    output_heads[0] only, backbone and every other species head frozen -- mirrors
    TwinModel/finetune.py exactly, unlike run_single_step (which assumes train.py's
    from-scratch, fully-trainable scenario and is the wrong comparison for an actually
    fine-tuned checkpoint). pt_twin_model must be a TwinModel wrapping the converted
    base model -- its train() override already does the base_model.eval() /
    output_heads[0].train() split this needs. On the Keras side no special-casing is
    required: Keras forces any non-trainable BatchNormalization layer into inference
    mode (fixed running stats) regardless of the training=True argument, matching
    TwinModel's frozen backbone automatically -- see the finetune section in
    notes/implementation.md for why testing this against test_tf_gradient_equivalence_real.py's
    blanket-trainable assumption instead produced a large, spurious mismatch.

    species_order should be load_pretrained's checkpoint args["species_order"]; per
    TwinModel's own convention, output_heads[0]/species_order[0] is always the one
    left trainable.
    """
    force_tf_cpu()
    import tensorflow as tf
    import tf_keras as keras
    import torch

    from promoterai_torch.utils import clip_grad_norm_per_parameter

    species = species_order or ("human",)

    def head0(pred):
        pred_list, _ = normalize_keras_outputs(pred, species_order)
        return pred_list[0]

    # --- Keras: one manual finetuning step ---
    keras_optimizer = keras.optimizers.AdamW(
        learning_rate=lr, weight_decay=wd, epsilon=eps, clipnorm=clip_norm
    )
    x_ref_tf = tf.constant(x_ref_np)
    x_alt_tf = tf.constant(x_alt_np)
    y_tf = tf.constant(y_np)
    bn_stats_keras_before = {
        v.name.rstrip(":0"): v.numpy().copy()
        for v in keras_model.non_trainable_variables
        if "moving_" in v.name
    }
    with tf.GradientTape() as tape:
        pred_ref = head0(keras_model(x_ref_tf, training=True))
        pred_alt = head0(keras_model(x_alt_tf, training=True))
        diff_keras = tf.reduce_mean(pred_alt - pred_ref, axis=[1, 2])
        keras_loss = tf.reduce_mean(tf.square(diff_keras - y_tf))

    def mapped(tensors):
        return keras_tensors_to_pt(tensors, num_blocks, shortcut_nums_desc, species)

    keras_vars = keras_model.trainable_variables
    keras_names = [v.name.rstrip(":0") for v in keras_vars]
    raw_grads = tape.gradient(keras_loss, keras_vars)
    clipped_grads = [tf.clip_by_norm(g, clip_norm) for g in raw_grads]
    params_before = {n: v.numpy().copy() for n, v in zip(keras_names, keras_vars)}
    keras_optimizer.apply_gradients(zip(clipped_grads, keras_vars))
    params_after = {n: v.numpy().copy() for n, v in zip(keras_names, keras_vars)}
    bn_stats_keras_after = {
        v.name.rstrip(":0"): v.numpy().copy()
        for v in keras_model.non_trainable_variables
        if "moving_" in v.name
    }
    raw_grads_keras = {n: g.numpy() for n, g in zip(keras_names, raw_grads)}
    clipped_grads_keras = {n: g.numpy() for n, g in zip(keras_names, clipped_grads)}
    param_deltas_keras = {n: params_after[n] - params_before[n] for n in params_before}

    keras_tiny_vars = [tf.Variable(params_before[n].copy()) for n in keras_names]
    keras.optimizers.AdamW(
        learning_rate=lr, weight_decay=wd, epsilon=tiny_eps, clipnorm=clip_norm
    ).apply_gradients(zip(clipped_grads, keras_tiny_vars))
    deltas_keras_tiny = {
        n: v.numpy() - params_before[n] for n, v in zip(keras_names, keras_tiny_vars)
    }

    # --- PyTorch: one manual finetuning step, same batch, same (converted) weights ---
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch_device = torch.device(device)
    pt_twin_model.to(torch_device)
    pt_twin_model.train()  # TwinModel.train(): base_model.eval() + output_heads[0].train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, pt_twin_model.parameters()),
        lr=lr, weight_decay=wd, eps=eps,
    )
    x_ref_pt = torch.from_numpy(x_ref_np).to(torch_device)
    x_alt_pt = torch.from_numpy(x_alt_np).to(torch_device)
    y_pt = torch.from_numpy(y_np).to(torch_device)

    bn_stats_pt_before = _backbone_bn_stats(pt_twin_model.base_model)

    diff_pt = pt_twin_model(x_ref_pt, x_alt_pt)
    loss_pt = torch.mean((diff_pt - y_pt) ** 2)
    optimizer.zero_grad()
    loss_pt.backward()
    raw_grads_pt = {
        n.removeprefix("base_model."): p.grad.detach().clone()
        for n, p in pt_twin_model.named_parameters()
        if p.grad is not None
    }
    params_before_pt = {
        n.removeprefix("base_model."): p.detach().clone()
        for n, p in pt_twin_model.named_parameters()
        if p.requires_grad
    }
    clip_grad_norm_per_parameter(pt_twin_model.parameters(), max_norm=clip_norm)
    clipped_grads_pt = {
        n.removeprefix("base_model."): p.grad.detach().clone()
        for n, p in pt_twin_model.named_parameters()
        if p.grad is not None
    }
    optimizer.step()
    param_deltas_pt = {
        n.removeprefix("base_model."): (p.detach() - params_before_pt[n.removeprefix("base_model.")]).cpu().numpy()
        for n, p in pt_twin_model.named_parameters()
        if p.requires_grad
    }
    bn_stats_pt_after = _backbone_bn_stats(pt_twin_model.base_model)

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
        "bn_unchanged_keras": compare_all(bn_stats_keras_before, bn_stats_keras_after),
        "bn_unchanged_pt": compare_all(bn_stats_pt_before, bn_stats_pt_after),
    }
