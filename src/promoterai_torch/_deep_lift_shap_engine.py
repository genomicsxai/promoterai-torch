"""
PROTOTYPE: a memory-lean DeepLIFT/SHAP engine for PromoterAI.

Computes the same attributions as::

    tangermeme.deep_lift_shap.deep_lift_shap(wrapper, X, n_shuffles=K, random_state=s)

for ``wrapper(x) = model(x.transpose(1, 2))[head].mean(dim=(1, 2), keepdim)``, the
quantity examples/deepliftshap_region.py attributes, without tangermeme's hooks or
autograd. Ideas taken from Cherimoya's engine (cherimoya/_deep_lift_shap_engine.py):

* Joint rows. A step forwards each sequence once with k of its references,
  ``[x, r_1, ..., r_k]``, instead of tangermeme's k copies of ``x`` paired with
  ``r_1..r_k`` (2k rows).
* Sequence-half backward. tangermeme's rule for a sequence row reads the paired
  reference row's activations but never its gradient, and every PromoterAI op is
  row-wise, so the backward runs over the k (x, r_j) pairs only, not 2k rows.
* Hand-written backward with the rule in closed form. PromoterAI's only
  nonlinearities are ReLUs (stem, FFN, output heads), for which tangermeme uses the
  rescale rule: the upstream gradient times (relu(a_x) - relu(a_r)) / (a_x - a_r),
  or the ordinary ReLU gradient at a_x where |a_x - a_r| < 1e-6. BatchNorm (eval),
  the depthwise convolutions, the linear layers, residual adds and the head's
  stack/mean/crop are linear, so their ordinary transposes are exact.

PromoterAI-specific, for memory (the reason this exists: tangermeme keeps the full
autograd graph, ~26 GB per row at the published 24 x 1024 x 20480 scale, so not
even one sequence/reference pair fits a 12 GB GPU):

* Only each block's output is kept from the forward, (k + 1) x L x C per block.
  The backward recomputes one block at a time from its input to get the FFN's
  pre-ReLU activations (L x 4C per row, the dominant tensor), so peak memory is
  ~(n_blocks + 1) x (k + 1) x L x C plus one block's transients.
* References are processed in chunks of `ref_chunk`, so memory is bounded
  independently of n_shuffles; the sequence row is recomputed once per chunk.

Measured (examples/dls_engine_benchmark/benchmark_gpu.py; TITAN X Pascal, 12 GB;
24 blocks, L=20480, 1 sequence x 20 shuffles, random weights, float32):

    width  tangermeme (largest batch that fits)   engine ref_chunk=1   engine fastest
      64   3.7 s, 5.92 GiB (bs=1)                  3.1 s, 0.44 GiB      2.0 s, 5.37 GiB (chunk 20)
     128   6.7 s, 9.75 GiB (bs=1)                  5.9 s, 0.87 GiB      4.1 s, 5.52 GiB (chunk 10)
     256   out of memory at bs=1                  13.4 s, 1.75 GiB     10.1 s, 4.81 GiB (chunk 4)

Attributions agree with tangermeme to ~1e-7 relative L2 in float32 and to ~1e-16
in float64 (tests/test_deep_lift_shap_engine.py). Memory grows linearly in width at a
fixed ref_chunk, which extrapolates to ~7 GiB at the published width 1024 with
ref_chunk=1 -- unverified, since the full model was not run.

Not yet: torch.compile of the passes, a memory planner choosing `ref_chunk`,
batching several sequences per step, targets other than the head mean, or a
public API / CLI.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.deep_lift_shap import hypothetical_attributions

from promoterai_torch.architecture import MetaFormerBlock, PromoterAI

_RESCALE_EPS = 1e-6  # tangermeme's threshold for falling back to the plain gradient


def _rescale(g, a_x, a_r):
    """tangermeme's rescale rule through a ReLU for the sequence half of each pair.

    g: upstream gradient (k, ...) at the ReLU output; a_x (1, ...) the sequence
    row's ReLU input and a_r (k, ...) the references'. Returns the gradient at the
    ReLU input, (k, ...).
    """
    delta = a_x - a_r
    secant = (torch.relu(a_x) - torch.relu(a_r)) / torch.where(
        delta.abs() < _RESCALE_EPS, torch.ones_like(delta), delta
    )
    plain = (a_x > 0).to(g.dtype)
    return g * torch.where(delta.abs() < _RESCALE_EPS, plain, secant)


def _bn_scale(bn: torch.nn.BatchNorm1d) -> torch.Tensor:
    """Per-channel multiplier of an eval-mode BatchNorm (its transpose is the same scale)."""
    return bn.weight / torch.sqrt(bn.running_var + bn.eps)


def _block_pre_relu(block: MetaFormerBlock, h: torch.Tensor) -> torch.Tensor:
    """The FFN's pre-ReLU activations ffn1(bn2(intermediate)) for block input h (R, L, C)."""
    conv = block.dw_conv(block.bn1(h.transpose(1, 2))).transpose(1, 2)
    inter = h + conv
    return block.ffn1(block.bn2(inter.transpose(1, 2)).transpose(1, 2))


def _block_backward(block: MetaFormerBlock, h: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """DeepLIFT multipliers at a block's input from those at its output.

    h: the block's input for the joint rows (1 + k, L, C), sequence first; G: (k, L, C)
    multipliers at the block output for the k pairs. Returns (k, L, C).
    """
    u = _block_pre_relu(block, h)
    g = _rescale(G @ block.ffn2.weight, u[:1], u[1:])  # (k, L, 4C)
    del u
    g_inter = G + (g @ block.ffn1.weight) * _bn_scale(block.bn2)
    # Transpose of the 'same'-padded depthwise conv: the same conv with the taps reversed.
    d = block.dw_conv.dilation[0]
    g_conv = F.conv1d(
        g_inter.transpose(1, 2),
        block.dw_conv.weight.flip(-1),
        padding="same",
        dilation=d,
        groups=block.dw_conv.groups,
    ).transpose(1, 2)
    return g_inter + g_conv * _bn_scale(block.bn1)


@torch.no_grad()
def _pairs(model: PromoterAI, rows: torch.Tensor, head: int):
    """Forward the joint rows [x, r_1..r_k] (1 + k, L, 4) and backpropagate the rule.

    Returns (multipliers (k, L, 4), y (1 + k,)).
    """
    k = rows.shape[0] - 1
    out_head = model.output_heads[head]
    L = rows.shape[1]
    crop = out_head.output_crop // 2 if out_head.output_crop > 0 else 0
    lo, hi = crop, L - crop
    n_proj = len(out_head.projections)
    D = out_head.projections[0].out_features
    c = 1.0 / (n_proj * (hi - lo) * D)  # d y / d (each ReLU output in the crop)

    # Forward, keeping the stem's pre-ReLU activations and every block output.
    z = model.stem(rows.transpose(1, 2)).transpose(1, 2)  # (R, L, C)
    layers = [torch.relu(z)]
    for block in model.blocks:
        layers.append(block(layers[-1]))

    # Output and the head's contribution to the multipliers at each shortcut layer.
    y = torch.zeros(k + 1, dtype=rows.dtype, device=rows.device)
    head_grads: dict[int, torch.Tensor] = {}
    for act, proj, i in zip(out_head.acts, out_head.projections, out_head.shortcut_indices):
        v = proj(layers[i][:, lo:hi])  # (R, Lout, D)
        y += act(v).sum(dim=(1, 2)) * c
        g = _rescale(torch.full_like(v[1:], c), v[:1], v[1:]) @ proj.weight  # (k, Lout, C)
        head_grads[i] = head_grads.get(i, 0) + g
        del v

    G = torch.zeros(k, L, layers[0].shape[-1], dtype=rows.dtype, device=rows.device)
    for b in range(len(model.blocks), 0, -1):
        if b in head_grads:
            G[:, lo:hi] += head_grads.pop(b)
        G = _block_backward(model.blocks[b - 1], layers[b - 1], G)
        layers[b] = None  # free as we go
    if 0 in head_grads:
        G[:, lo:hi] += head_grads.pop(0)

    G = _rescale(G, z[:1], z[1:])  # stem ReLU
    m = G @ model.stem.weight[:, :, 0]  # (k, L, 4): stem is a kernel-1 conv
    return m, y


def deep_lift_shap(
    model: PromoterAI,
    X: torch.Tensor,
    head: int = 0,
    n_shuffles: int = 20,
    references: torch.Tensor | None = None,
    random_state: int | None = None,
    ref_chunk: int = 4,
    hypothetical: bool = False,
    return_deltas: bool = False,
    device: str | torch.device | None = None,
):
    """DeepLIFT/SHAP attributions of mean(model(x)[head]) for one-hot X (N, 4, L).

    Matches tangermeme's `deep_lift_shap` on the wrapper above, including its
    references: dinucleotide shuffles drawn with seed `random_state + j` for the
    j-th reference, or `references` (N, n_shuffles, 4, L) if given. Returns
    attributions (N, 4, L) (times X unless `hypothetical`), plus convergence
    deltas (N, n_shuffles) if `return_deltas`.
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model = model.eval()
    N, _, L = X.shape
    attrs, deltas = [], []
    for i in range(N):
        x = X[i : i + 1].to(dtype)
        if references is not None:
            refs = references[i].to(dtype)
            K = refs.shape[0]
        else:
            K = n_shuffles
            refs = torch.cat(
                [
                    dinucleotide_shuffle(
                        X[i : i + 1], n=1, random_state=None if random_state is None else random_state + j
                    )[:, 0]
                    for j in range(K)
                ]
            ).to(dtype)
        hyp_sum = torch.zeros(1, 4, L, dtype=dtype)
        row_deltas = []
        for s in range(0, K, ref_chunk):
            r = refs[s : s + ref_chunk]
            rows = torch.cat([x, r]).transpose(1, 2).to(device)  # (1 + k, L, 4)
            m, y = _pairs(model, rows, head)
            m = m.transpose(1, 2).cpu()  # (k, 4, L), tangermeme's layout
            y = y.cpu()
            row_deltas.append(((y[0] - y[1:]) - ((x - r) * m).sum(dim=(1, 2))).abs())
            (hyp,) = hypothetical_attributions((m,), (x.expand_as(r),), (r,))
            hyp_sum += hyp.sum(dim=0, keepdim=True)
        attr = hyp_sum / K
        attrs.append(attr if hypothetical else attr * x)
        deltas.append(torch.cat(row_deltas))
    attrs = torch.cat(attrs)
    return (attrs, torch.stack(deltas)) if return_deltas else attrs
