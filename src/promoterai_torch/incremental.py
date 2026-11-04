"""
Exact, receptive-field-limited variant scoring for same-length substitutions.

Every PromoterAI operation is either position-wise (the kernel-1 stem, eval-mode
BatchNorm, the FFN, the head projections and their ReLUs) or a kernel-5 dilated
depthwise convolution. An edit at positions [lo, hi) therefore changes block b's
output only on [lo - 2 d_0 - ... - 2 d_b, hi + 2 d_0 + ... + 2 d_b), where d_i is
block i's dilation, and leaves every other position equal to the reference's.
Summed over the 24 published blocks, those windows cover ~12% of the positions a
full forward pass computes.

`TwinModel` scores a variant as mean(head(alt) - head(ref)) over output positions
and tracks. Outside the changed window that difference is exactly zero, so the
score can be computed by running the reference once (caching every block's
output), then, for each alt, recomputing only the changed window of each block
from the cached reference activations around it. This is the fastISM idea (Nair
et al., Bioinformatics 2022) applied to PromoterAI's architecture, and the result
matches `TwinModel` up to floating-point rounding.

Alts that share a reference window (e.g. the three SNVs at a site) share its
cached forward pass, which is where most of the speedup comes from: three SNVs at
one site cost one full pass plus three ~12% passes instead of six full passes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from promoterai_torch.architecture import MetaFormerBlock, PromoterAI


@torch.no_grad()
def reference_layers(model: PromoterAI, x_ref: torch.Tensor) -> list[torch.Tensor]:
    """Full forward of reference sequences (B, L, 4), returning [layer_0, ..., layer_N].

    layer_0 is the stem output and layer_i the output of block i - 1, each
    (B, L, model_dim). The model must be in eval mode (BatchNorm uses running
    statistics, which is what makes it position-wise).
    """
    h = model.stem_act(model.stem(x_ref.transpose(1, 2))).transpose(1, 2)
    layers = [h]
    for block in model.blocks:
        h = block(h)
        layers.append(h)
    return layers


def edit_span(x_ref: torch.Tensor, x_alt: torch.Tensor) -> tuple[int, int] | None:
    """[lo, hi) of the positions where any alt (B, L, 4) differs from x_ref (1 or B, L, 4).

    None when no alt differs (e.g. a skipped variant's all-zero placeholder pair).
    """
    changed = (x_ref != x_alt).any(dim=-1).any(dim=0).nonzero()
    if len(changed) == 0:
        return None
    return int(changed[0]), int(changed[-1]) + 1


def _block_window(
    block: MetaFormerBlock,
    ref_in: torch.Tensor,
    lo: int,
    hi: int,
    alt_in: torch.Tensor,
) -> tuple[int, int, torch.Tensor]:
    """One block's output for the alts on the window their edit can reach.

    ref_in is the block's cached reference input (1, L, C); alt_in the alts'
    input on [lo, hi), (B, hi - lo, C), equal to ref_in everywhere else.
    Returns (lo_out, hi_out, alt_out) with alt_out (B, hi_out - lo_out, C).
    """
    length = ref_in.shape[1]
    d = block.dw_conv.dilation[0]
    reach = (block.dw_conv.kernel_size[0] - 1) // 2 * d
    lo_out, hi_out = max(0, lo - reach), min(length, hi + reach)
    lo_need, hi_need = lo_out - reach, hi_out + reach  # may extend past [0, length)
    a, b = max(0, lo_need), min(length, hi_need)

    seg = ref_in[:, a:b].expand(alt_in.shape[0], -1, -1).clone()
    seg[:, lo - a : hi - a] = alt_in
    # padding="same" zero-pads the conv's input (bn1's output) at the true
    # sequence ends, so pad here only where the window runs past them.
    z = F.pad(block.bn1(seg.transpose(1, 2)), (a - lo_need, hi_need - b))
    conv = F.conv1d(
        z, block.dw_conv.weight, block.dw_conv.bias, dilation=d, groups=block.dw_conv.groups
    )  # "valid" -> exactly [lo_out, hi_out)
    intermediate = seg[:, lo_out - a : hi_out - a] + conv.transpose(1, 2)
    t = block.bn2(intermediate.transpose(1, 2)).transpose(1, 2)
    return lo_out, hi_out, intermediate + block.ffn2(block.act(block.ffn1(t)))


@torch.no_grad()
def substitution_diffs(
    model: PromoterAI,
    ref_layers: list[torch.Tensor],
    x_alt: torch.Tensor,
    span: tuple[int, int],
) -> torch.Tensor:
    """TwinModel's mean(head_0(alt) - head_0(ref)) for alts sharing one reference.

    ref_layers: `reference_layers` of one reference, each (1, L, C).
    x_alt: (B, L, 4) alts that differ from the reference only on span = [lo, hi).
    Returns (B,), equal to `TwinModel(model)(x_ref, x_alt)` up to rounding.
    """
    lo, hi = span
    alt = model.stem_act(model.stem(x_alt[:, lo:hi].transpose(1, 2))).transpose(1, 2)
    windows = [(lo, hi, alt)]
    for i, block in enumerate(model.blocks):
        lo, hi, alt = _block_window(block, ref_layers[i], lo, hi, alt)
        windows.append((lo, hi, alt))

    head = model.output_heads[0]
    length = x_alt.shape[1]
    crop = head.output_crop // 2 if head.output_crop > 0 else 0
    out_lo, out_hi = crop, length - crop
    total = torch.zeros(x_alt.shape[0], device=x_alt.device, dtype=alt.dtype)
    for act, proj, i in zip(head.acts, head.projections, head.shortcut_indices):
        w_lo, w_hi, w_alt = windows[i]
        c_lo, c_hi = max(w_lo, out_lo), min(w_hi, out_hi)
        if c_lo >= c_hi:
            continue  # this shortcut's change lies entirely in the cropped margin
        ref = ref_layers[i][:, c_lo:c_hi]
        w_alt = w_alt[:, c_lo - w_lo : c_hi - w_lo]
        total += (act(proj(w_alt)) - act(proj(ref))).sum(dim=(1, 2))
    n = len(head.projections) * (out_hi - out_lo) * head.projections[0].out_features
    return total / n
