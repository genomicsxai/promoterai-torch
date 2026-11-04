"""
A memory-lean DeepLIFT/SHAP engine for PromoterAI.

Computes the same attributions as::

    tangermeme.deep_lift_shap.deep_lift_shap(wrapper, X, n_shuffles=K, random_state=s)

for ``wrapper(x) = model(x.transpose(1, 2))[head][..., tracks].mean()`` per sequence
(the quantity examples/deepliftshap_region.py attributes when ``tracks`` is all of
them), without tangermeme's hooks or autograd. Ideas taken from Cherimoya's engine
(cherimoya/_deep_lift_shap_engine.py):

* Joint rows. tangermeme forwards k copies of ``x`` paired with ``r_1..r_k`` (2k
  rows); here each sequence is forwarded once and its forward is cached across
  reference chunks, so a chunk forwards only its k references.
* Sequence-half backward. tangermeme's rule for a sequence row reads the paired
  reference row's activations but never its gradient, and every PromoterAI op is
  row-wise, so the backward runs over the k (x, r_j) pairs only, not 2k rows.
* Hand-written backward with the rule in closed form. PromoterAI's only
  nonlinearities are ReLUs (stem, FFN, output heads), for which tangermeme uses the
  rescale rule: the upstream gradient times (relu(a_x) - relu(a_r)) / (a_x - a_r),
  or the ordinary ReLU gradient at a_x where |a_x - a_r| < 1e-6. BatchNorm (eval),
  the depthwise convolutions, the linear layers, residual adds and the head's
  stack/mean/crop are linear, so their ordinary transposes are exact.

PromoterAI-specific, for memory (tangermeme keeps the full autograd graph, ~26 GB
per row at the published 24 x 1024 x 20480 scale, so not even one
sequence/reference pair fits a 12 GB GPU):

* Only each block's output is kept from the forward, L x C per row per block. The
  backward recomputes one block at a time from its input to get the FFN's pre-ReLU
  activations (L x 4C per row, the dominant tensor) and computes the heads'
  contributions when it reaches each shortcut layer, freeing layers as it goes.
* References are processed in chunks, so memory is bounded independently of
  n_shuffles. `_plan` picks the chunk (and, once all of a sequence's references
  fit, how many sequences to batch per step) from a memory model and the free
  device memory; a step that still runs out of memory is retried at half the size.
"""

from __future__ import annotations

import os
import warnings

import torch
import torch.nn.functional as F
from tangermeme.ersatz import dinucleotide_shuffle

from promoterai_torch.architecture import MetaFormerBlock, PromoterAI

_RESCALE_EPS = 1e-6  # tangermeme's threshold for falling back to the plain gradient


# --------------------------------------------------------------------------- rules


def _rescale(g, a_x, a_r, eps: float = _RESCALE_EPS):
    """tangermeme's rescale rule through a ReLU for the sequence half of each pair.

    g: upstream gradient (B, k, ...) at the ReLU output; a_x (B, 1, ...) the
    sequences' ReLU inputs and a_r (B, k, ...) their references'. Returns the
    gradient at the ReLU input, (B, k, ...). Functional, for torch.compile to fuse.
    """
    delta = a_x - a_r
    small = delta.abs() < eps
    secant = (torch.relu(a_x) - torch.relu(a_r)) / torch.where(small, torch.ones_like(delta), delta)
    return g * torch.where(small, (a_x > 0).to(g.dtype), secant)


def _rescale_(g, a_x, a_r):
    """`_rescale` in place, for eager mode: overwrites g (the result) and a_r."""
    small = (a_x - a_r).abs_() < _RESCALE_EPS
    delta = a_r.neg().add_(a_x)
    ratio = a_r.clamp_(min=0).neg_().add_(a_x.clamp(min=0))  # relu(a_x) - relu(a_r)
    ratio.div_(delta.masked_fill_(small, 1.0))
    del delta
    torch.where(small, (a_x > 0).to(ratio.dtype), ratio, out=ratio)
    return g.mul_(ratio)


def _bn_scale(bn: torch.nn.BatchNorm1d) -> torch.Tensor:
    """Per-channel multiplier of an eval-mode BatchNorm (its transpose is the same scale)."""
    return bn.weight / torch.sqrt(bn.running_var + bn.eps)


def _block_pre_relu(block: MetaFormerBlock, h: torch.Tensor) -> torch.Tensor:
    """The FFN's pre-ReLU activations ffn1(bn2(intermediate)) for block input h (R, L, C)."""
    conv = block.dw_conv(block.bn1(h.transpose(1, 2))).transpose(1, 2)
    inter = h + conv
    return block.ffn1(block.bn2(inter.transpose(1, 2)).transpose(1, 2))


def _block_backward(block, h_x, h_r, G, rescale):
    """DeepLIFT multipliers at a block's input from those at its output.

    h_x (B, L, C) and h_r (B, k, L, C): the block's input for the sequences and their
    references; G (B, k, L, C): multipliers at the block output. Returns (B, k, L, C).
    """
    B, k = G.shape[:2]
    u_x = _block_pre_relu(block, h_x).unsqueeze(1)  # (B, 1, L, 4C)
    u_r = _block_pre_relu(block, h_r.flatten(0, 1)).unflatten(0, (B, k))
    g = rescale(G @ block.ffn2.weight, u_x, u_r)  # (B, k, L, 4C)
    del u_x, u_r
    g_inter = torch.addcmul(G, g @ block.ffn1.weight, _bn_scale(block.bn2))
    del g
    # Transpose of the 'same'-padded depthwise conv: the same conv with the taps reversed.
    g_conv = F.conv1d(
        g_inter.flatten(0, 1).transpose(1, 2),
        block.dw_conv.weight.flip(-1),
        padding="same",
        dilation=block.dw_conv.dilation[0],
        groups=block.dw_conv.groups,
    ).transpose(1, 2)
    return torch.addcmul(g_inter, g_conv.unflatten(0, (B, k)), _bn_scale(block.bn1))


# --------------------------------------------------------------------- passes


def _forward(model: PromoterAI, rows: torch.Tensor):
    """Stem pre-ReLU activations and every block output for rows (R, L, 4)."""
    z = model.stem(rows.transpose(1, 2)).transpose(1, 2)  # (R, L, C)
    layers = [torch.relu(z)]
    for block in model.blocks:
        layers.append(block(layers[-1]))
    return z, layers


def _head_weights(model, head, tracks, L, dtype, device):
    """(lo, hi, w): the output crop and per-track weights w (D,) such that the
    attributed output is the sum over shortcut projections, cropped positions and
    tracks of w * relu(projection) -- the mean over the selected tracks."""
    out_head = model.output_heads[head]
    crop = out_head.output_crop // 2 if out_head.output_crop > 0 else 0
    lo, hi = crop, L - crop
    D = out_head.projections[0].out_features
    idx = torch.arange(D) if tracks is None else torch.as_tensor(tracks, dtype=torch.long).flatten()
    if len(idx) == 0 or len(torch.unique(idx)) != len(idx) or int(idx.min()) < -D or int(idx.max()) >= D:
        raise ValueError(f"tracks must be distinct indices into head {head}'s {D} tracks")
    w = torch.zeros(D, dtype=dtype)
    w[idx] = 1.0 / (len(out_head.projections) * (hi - lo) * len(idx))
    return lo, hi, w.to(device)


def _output(model, head, layers, lo, hi, w):
    """The attributed output per row, from a forward's layers."""
    out_head = model.output_heads[head]
    y = 0
    for proj, i in zip(out_head.projections, out_head.shortcut_indices):
        y = y + (torch.relu(proj(layers[i][:, lo:hi])) * w).sum(dim=(1, 2))
    return y


def _step(model, head, lo, hi, w, z_x, layers_x, refs, fns):
    """Multipliers (B, k, L, 4) and outputs (B, k) for references refs (B, k, L, 4) of
    the sequences whose cached forward is z_x, layers_x (each (B, L, C))."""
    B, k = refs.shape[:2]
    out_head = model.output_heads[head]
    shortcut = dict(zip(out_head.shortcut_indices, out_head.projections))
    rescale = fns["rescale"]
    z_r, layers_r = fns["forward"](model, refs.flatten(0, 1))
    G = torch.zeros(B, k, *layers_x[0].shape[1:], dtype=refs.dtype, device=refs.device)
    y_r = torch.zeros(B, k, dtype=refs.dtype, device=refs.device)
    for b in range(len(model.blocks), -1, -1):
        if b in shortcut:
            proj = shortcut[b]
            v_x = proj(layers_x[b][:, lo:hi]).unsqueeze(1)  # (B, 1, Lout, D)
            v_r = proj(layers_r[b][:, lo:hi]).unflatten(0, (B, k))
            y_r += (torch.relu(v_r) * w).sum(dim=(2, 3))
            g = rescale(w.expand_as(v_r).clone(), v_x, v_r)
            G[:, :, lo:hi] += g @ proj.weight
            del v_x, v_r, g
        if b == 0:
            break
        layers_r[b] = None  # the backward below needs only block b's input
        G = fns["block_backward"](
            model.blocks[b - 1], layers_x[b - 1], layers_r[b - 1].unflatten(0, (B, k)), G, rescale
        )
    G = rescale(G, z_x.unsqueeze(1), z_r.unflatten(0, (B, k)))  # stem ReLU
    return G @ model.stem.weight[:, :, 0], y_r  # stem is a kernel-1 conv


def _attribute(model, head, lo, hi, w, x_rows, r_rows, k, fns):
    """Summed hypothetical attributions (B, L, 4) and deltas (B, K) for sequences
    x_rows (B, L, 4) and their references r_rows (B, K, L, 4), k references a step."""
    K = r_rows.shape[1]
    z_x, layers_x = fns["forward"](model, x_rows)
    y_x = _output(model, head, layers_x, lo, hi, w)  # (B,)
    hyp_sum = torch.zeros_like(x_rows)
    deltas = torch.empty(x_rows.shape[0], K, dtype=x_rows.dtype, device=x_rows.device)
    for s in range(0, K, k):
        r = r_rows[:, s : s + k]
        m, y_r = _step(model, head, lo, hi, w, z_x, layers_x, r, fns)  # (B, k, L, 4)
        # tangermeme's hypothetical attributions: the multipliers projected onto each
        # base's difference from the reference, m_i - sum_c r_c m_c.
        hyp_sum += (m - (r * m).sum(-1, keepdim=True)).sum(1)
        lin = ((x_rows.unsqueeze(1) - r) * m).sum(dim=(2, 3))
        deltas[:, s : s + k] = ((y_x[:, None] - y_r) - lin).abs()
    return hyp_sum, deltas


# --------------------------------------------------------------------- planning


def _available_memory(device: torch.device) -> int:
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        # Memory PyTorch has cached but isn't using is reusable by this process too.
        return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 8 << 30


# Activations, in units of one (L, C) tensor, that `_attribute` holds at its peak per
# sequence (cached forward + pre-ReLU recompute) and per reference (forward, G and the
# block backward's transients). Fitted on a TITAN X; see `_plan`.
_PER_SEQ_EXTRA, _PER_REF_EXTRA = 13, 16


def _plan(model, L, n_seqs, n_refs, dtype, device, max_memory=None):
    """(batch_size, ref_chunk) that fit in max_memory bytes (default 85% of free).

    A sequence costs ~(n_blocks + 13) (L, C) activations: its cached forward
    (n_blocks + 2 incl. the stem's pre-ReLU), its pre-ReLU recompute (4C wide) and
    head projections. A reference costs ~(n_blocks + 16): its forward, G, its
    pre-ReLU recompute, the rescaled gradient, the rule's temporaries and the FFN/conv
    transposes. Fitted to peak CUDA memory at 24 blocks x widths 64-1024 x L=20480
    (measured ~n_blocks + 10 and + 15) on a TITAN X. The
    reference chunk is the largest that fits beside one sequence; only when all of a
    sequence's references fit are several sequences batched.
    """
    if max_memory is None:
        max_memory = int(0.85 * _available_memory(device))
    C = model.stem.out_channels
    M = L * C * torch.tensor([], dtype=dtype).element_size()
    nb = len(model.blocks)
    per_seq, per_ref = (nb + _PER_SEQ_EXTRA) * M, (nb + _PER_REF_EXTRA) * M
    k = int((max_memory - per_seq) // per_ref)
    if k < 1:
        warnings.warn(
            f"one sequence and one reference need ~{(per_seq + per_ref) / 2**30:.1f} GiB, "
            f"more than the {max_memory / 2**30:.1f} GiB budget; trying anyway",
            stacklevel=3,
        )
        return 1, 1
    if k < n_refs:
        return 1, k
    B = int(max_memory // (per_seq + n_refs * per_ref))
    return max(1, min(B, n_seqs)), n_refs


# --------------------------------------------------------------------- driver


def _shuffles(x, n, random_state):
    """tangermeme's references for x (1, 4, L): dinucleotide shuffles seeded random_state + j."""
    return torch.cat(
        [
            dinucleotide_shuffle(x, n=1, random_state=None if random_state is None else random_state + j)[:, 0]
            for j in range(n)
        ]
    )


def _passes(compile: bool, device: torch.device):
    """The forward, block-backward and rescale functions to run.

    Eagerly, the rescale rule runs in place (fewer temporaries). With `compile`,
    torch.compile fuses the passes where its Triton backend can run (CPU, or CUDA
    capability >= 7.0); on older GPUs (e.g. Pascal), where Triton can't, the rescale
    rule -- the bulk of the elementwise work -- is fused by TorchScript's NNC fuser
    instead, which generates CUDA through NVRTC and needs no Triton.
    """
    eager = {"rescale": _rescale_, "forward": _forward, "block_backward": _block_backward}
    if not compile:
        return eager
    if device.type == "cuda" and torch.cuda.get_device_capability(device) < (7, 0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # TorchScript's deprecation notice
            return {**eager, "rescale": torch.jit.script(_rescale)}
    return {
        "rescale": torch.compile(_rescale, dynamic=True),
        "forward": torch.compile(_forward, dynamic=True),
        "block_backward": torch.compile(_block_backward, dynamic=True),
    }


@torch.no_grad()
def deep_lift_shap(
    model: PromoterAI,
    X: torch.Tensor,
    head: int = 0,
    tracks=None,
    n_shuffles: int = 20,
    references: torch.Tensor | None = None,
    random_state: int | None = None,
    batch_size: int | None = None,
    ref_chunk: int | None = None,
    max_memory: int | None = None,
    hypothetical: bool = False,
    return_deltas: bool = False,
    compile: bool = False,
    device: str | torch.device | None = None,
):
    """DeepLIFT/SHAP attributions for one-hot X (N, 4, L) (tangermeme's layout).

    The attributed output is the mean of head `head`'s prediction over its output
    window and over `tracks` (indices into the head's tracks; default all).
    References are dinucleotide shuffles drawn with seed `random_state + j` for the
    j-th, as tangermeme does, or `references` (N, n_refs, 4, L).

    `batch_size` sequences are processed per step, each with `ref_chunk` of its
    references; both default to what fits in `max_memory` bytes (default 85% of the
    device's free memory), and a step that runs out of memory is retried at half the
    size. `compile=True` runs the passes through torch.compile (needs a CPU or a
    CUDA GPU of capability >= 7.0; otherwise warns and runs eagerly).

    Returns attributions (N, 4, L) (times X unless `hypothetical`) on the CPU, plus
    convergence deltas |sum(attr) - (y(x) - y(ref))| (N, n_refs) if `return_deltas`.
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model = model.eval().to(device)
    N, _, L = X.shape
    K = references.shape[1] if references is not None else n_shuffles
    plan_B, plan_k = _plan(model, L, N, K, dtype, device, max_memory)
    k = min(ref_chunk or plan_k, K)
    B = batch_size or (plan_B if k == K else 1)
    lo, hi, w = _head_weights(model, head, tracks, L, dtype, device)
    fns = _passes(compile, device)
    with torch._dynamo.config.patch(recompile_limit=64, accumulated_recompile_limit=4096):
        return _run(model, X, head, lo, hi, w, references, K, random_state, B, k, fns, device, dtype,
                    hypothetical, return_deltas)


def _run(model, X, head, lo, hi, w, references, K, random_state, B, k, fns, device, dtype,
         hypothetical, return_deltas):
    N, _, L = X.shape

    attrs = torch.empty(N, 4, L, dtype=dtype)
    deltas = torch.empty(N, K, dtype=dtype)
    i = 0
    while i < N:
        b = min(B, N - i)
        Xb = X[i : i + b].to(dtype)
        if references is not None:
            refs = references[i : i + b].to(dtype)
        else:
            refs = torch.stack([_shuffles(X[j : j + 1], K, random_state) for j in range(i, i + b)]).to(dtype)
        x_rows = Xb.to(device).transpose(1, 2).contiguous()  # (b, L, 4)
        r_rows = refs.to(device).transpose(2, 3).contiguous()  # (b, K, L, 4)
        oom = False
        try:
            hyp_sum, d = _attribute(model, head, lo, hi, w, x_rows, r_rows, k, fns)
        except torch.OutOfMemoryError:
            oom = True  # retry outside the handler, once the traceback's tensors are freed
        if oom:
            if b == 1 and k == 1:
                raise torch.OutOfMemoryError(
                    "out of memory at batch_size=1, ref_chunk=1; use a smaller model, a "
                    "shorter sequence or a larger device"
                )
            B, k = (max(1, b // 2), k) if b > 1 else (1, max(1, k // 2))
            warnings.warn(f"out of memory; retrying with batch_size={B}, ref_chunk={k}", stacklevel=2)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        attr = (hyp_sum / K).transpose(1, 2).cpu()
        attrs[i : i + b] = attr if hypothetical else attr * Xb
        deltas[i : i + b] = d.cpu()
        del hyp_sum, d
        i += b
    return (attrs, deltas) if return_deltas else attrs
