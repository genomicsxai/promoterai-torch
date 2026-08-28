"""Fused Triton kernel for depthwise dilated 1D convolution ("same" padding).

PyTorch's grouped/depthwise Conv1d falls back to a generic cuDNN grouped-convolution
path that's memory-bound and poorly specialized for the many-channels/small-kernel
shape used by MetaFormerBlock's token mixer. This module implements the same
"shift-and-accumulate" depthwise conv directly in Triton, avoiding that generic path.

Triton kernel compilation is only reliable on newer NVIDIA GPUs (Volta/sm_70+), so
callers must fall back to nn.Conv1d when this isn't available -- see
`triton_dw_conv_supported`.
"""

from __future__ import annotations

from functools import cache

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only without triton installed
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc


# Triton codegen is unreliable on pre-Volta GPUs; require sm_70+.
_MIN_CUDA_CAPABILITY = (7, 0)


def triton_available() -> bool:
    """Return True if the triton package could be imported."""
    return triton is not None


@cache
def triton_dw_conv_supported(device_type: str, device_index: int | None) -> bool:
    """Return True if a Triton depthwise-conv kernel can run on this CUDA device."""
    if not triton_available() or device_type != "cuda" or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(device_index)
    return (major, minor) >= _MIN_CUDA_CAPABILITY


def _autotune_configs():
    configs = []
    for num_warps in (4, 8):
        for num_stages in (2, 3, 4):
            for block_l in (64, 128, 256):
                configs.append(
                    triton.Config(
                        {"BLOCK_L": block_l}, num_warps=num_warps, num_stages=num_stages
                    )
                )
    return configs


if triton is not None:

    @triton.autotune(configs=_autotune_configs(), key=["C", "L", "K"])
    @triton.jit
    def _dw_conv1d_fwd_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        y_ptr,
        stride_n,
        dilation,
        pad_left,
        L: tl.constexpr,
        C: tl.constexpr,
        K: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_L: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        # x, y: (N, C, L) contiguous. w: (C, K) contiguous.
        pid_n = tl.program_id(0)
        offs_c = tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)

        x_base = x_ptr + pid_n * stride_n
        y_base = y_ptr + pid_n * stride_n

        for l_start in tl.range(0, L, BLOCK_L):
            offs_l = l_start + tl.arange(0, BLOCK_L)
            mask_l = offs_l < L
            acc = tl.zeros((BLOCK_L, BLOCK_C), dtype=tl.float32)

            for k in tl.static_range(K):
                in_l = offs_l[:, None] + (k * dilation - pad_left)
                mask = mask_l[:, None] & mask_c[None, :] & (in_l >= 0) & (in_l < L)
                x_val = tl.load(
                    x_base + offs_c[None, :] * L + in_l, mask=mask, other=0.0
                ).to(tl.float32)
                w_val = tl.load(
                    w_ptr + offs_c[None, :] * K + k, mask=mask_c[None, :], other=0.0
                ).to(tl.float32)
                acc += x_val * w_val

            if HAS_BIAS:
                acc += bias[None, :]

            tl.store(
                y_base + offs_c[None, :] * L + offs_l[:, None],
                acc,
                mask=mask_l[:, None] & mask_c[None, :],
            )

    @triton.autotune(configs=_autotune_configs(), key=["C", "L", "K"])
    @triton.jit
    def _dw_conv1d_bwd_kernel(
        dy_ptr,
        x_ptr,
        w_ptr,
        dx_ptr,
        dw_ptr,
        dbias_ptr,
        stride_n,
        dilation,
        pad_left,
        L: tl.constexpr,
        C: tl.constexpr,
        K: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_L: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        # dy, x, dx: (N, C, L) contiguous. w: (C, K). dw: (N, K, C). dbias: (N, C).
        pid_n = tl.program_id(0)
        offs_c = tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        dy_base = dy_ptr + pid_n * stride_n
        x_base = x_ptr + pid_n * stride_n
        dx_base = dx_ptr + pid_n * stride_n

        # dX[m] = sum_k w[k] * dY[m + pad_left - k*dilation]
        for m_start in tl.range(0, L, BLOCK_L):
            offs_m = m_start + tl.arange(0, BLOCK_L)
            mask_m = offs_m < L
            dx_acc = tl.zeros((BLOCK_L, BLOCK_C), dtype=tl.float32)

            for k in tl.static_range(K):
                in_l = offs_m[:, None] + (pad_left - k * dilation)
                mask = mask_m[:, None] & mask_c[None, :] & (in_l >= 0) & (in_l < L)
                dy_val = tl.load(
                    dy_base + offs_c[None, :] * L + in_l, mask=mask, other=0.0
                ).to(tl.float32)
                w_val = tl.load(
                    w_ptr + offs_c[None, :] * K + k, mask=mask_c[None, :], other=0.0
                ).to(tl.float32)
                dx_acc += dy_val * w_val

            tl.store(
                dx_base + offs_c[None, :] * L + offs_m[:, None],
                dx_acc,
                mask=mask_m[:, None] & mask_c[None, :],
            )

        # dW[k] = sum_l dY[l] * X[l + k*dilation - pad_left]; dbias = sum_l dY[l]
        dw_acc = [tl.zeros((BLOCK_C,), dtype=tl.float32) for _ in range(K)]
        if HAS_BIAS:
            dbias_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

        for l_start in tl.range(0, L, BLOCK_L):
            offs_l = l_start + tl.arange(0, BLOCK_L)
            mask_l = offs_l < L
            dy_val = tl.load(
                dy_base + offs_c[None, :] * L + offs_l[:, None],
                mask=mask_l[:, None] & mask_c[None, :],
                other=0.0,
            ).to(tl.float32)

            if HAS_BIAS:
                dbias_acc += tl.sum(dy_val, axis=0)

            for k in tl.static_range(K):
                in_l = offs_l[:, None] + (k * dilation - pad_left)
                mask = mask_l[:, None] & mask_c[None, :] & (in_l >= 0) & (in_l < L)
                x_val = tl.load(
                    x_base + offs_c[None, :] * L + in_l, mask=mask, other=0.0
                ).to(tl.float32)
                dw_acc[k] += tl.sum(dy_val * x_val, axis=0)

        for k in tl.static_range(K):
            tl.store(dw_ptr + pid_n * (K * C) + k * C + offs_c, dw_acc[k], mask=mask_c)
        if HAS_BIAS:
            tl.store(dbias_ptr + pid_n * C + offs_c, dbias_acc, mask=mask_c)


class _DepthwiseDilatedConv1dFunction(torch.autograd.Function):
    """Autograd wrapper around the fused Triton depthwise dilated conv1d kernels."""

    @staticmethod
    def forward(ctx, x, weight2d, bias, dilation):
        # x: (N, C, L) contiguous. weight2d: (C, K) contiguous. bias: (C,) or None.
        N, C, L = x.shape
        K = weight2d.shape[1]
        pad_left = dilation * (K - 1) // 2
        block_c = triton.next_power_of_2(C)
        has_bias = bias is not None

        y = torch.empty_like(x)
        _dw_conv1d_fwd_kernel[(N,)](
            x,
            weight2d,
            bias if has_bias else weight2d,
            y,
            x.stride(0),
            dilation,
            pad_left,
            L=L,
            C=C,
            K=K,
            BLOCK_C=block_c,
            HAS_BIAS=has_bias,
        )

        ctx.save_for_backward(x, weight2d)
        ctx.dilation = dilation
        ctx.pad_left = pad_left
        ctx.has_bias = has_bias
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight2d = ctx.saved_tensors
        N, C, L = x.shape
        K = weight2d.shape[1]
        block_c = triton.next_power_of_2(C)
        dy = dy.contiguous()

        dx = torch.empty_like(x)
        dw_partial = torch.empty((N, K, C), device=x.device, dtype=torch.float32)
        dbias_partial = (
            torch.empty((N, C), device=x.device, dtype=torch.float32)
            if ctx.has_bias
            else None
        )

        _dw_conv1d_bwd_kernel[(N,)](
            dy,
            x,
            weight2d,
            dx,
            dw_partial,
            dbias_partial if ctx.has_bias else dw_partial,
            x.stride(0),
            ctx.dilation,
            ctx.pad_left,
            L=L,
            C=C,
            K=K,
            BLOCK_C=block_c,
            HAS_BIAS=ctx.has_bias,
        )

        dw = dw_partial.sum(dim=0).transpose(0, 1).contiguous().to(weight2d.dtype)
        dbias = dbias_partial.sum(dim=0).to(weight2d.dtype) if ctx.has_bias else None
        return dx.to(x.dtype), dw, dbias, None


def depthwise_dilated_conv1d_triton(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, dilation: int
) -> torch.Tensor:
    """Fused Triton depthwise dilated conv1d with "same" padding.

    x: (B, C, L). weight: (C, 1, K) as produced by nn.Conv1d(groups=C). bias: (C,) or None.
    Equivalent to nn.Conv1d(C, C, K, dilation=dilation, padding="same", groups=C)(x).
    """
    x = x.contiguous()
    weight2d = weight.squeeze(1).contiguous()
    return _DepthwiseDilatedConv1dFunction.apply(x, weight2d, bias, dilation)
