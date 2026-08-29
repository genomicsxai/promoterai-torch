"""
Time the fused Triton depthwise-dilated-conv1d kernel against nn.Conv1d on GPU.

Runs forward-only and forward+backward timing for both backends across the
dilation schedule MetaFormerBlock actually uses (see _dilation_rate in
architecture.py), at a given model width/kernel size/sequence length, and
prints a table with the speedup of Triton over plain torch.

Requires a CUDA GPU with triton installed (compute capability >= 7.0); exits
with an explanatory error otherwise.

Example:
    python examples/benchmark_dw_conv.py --model_dim 1024 --seq_len 20480
"""

from __future__ import annotations

import argparse
import time

import torch

from promoterai_torch.architecture import _dilation_rate
from promoterai_torch.triton_ops import (
    depthwise_dilated_conv1d_triton,
    triton_dw_conv_supported,
)


def _time_ms(fn, num_warmup: int, num_iters: int) -> float:
    """Return average wall time in ms for fn(), synchronizing around CUDA work."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / num_iters * 1000


def _benchmark_one(
    conv: torch.nn.Conv1d,
    x: torch.Tensor,
    dilation: int,
    num_warmup: int,
    num_iters: int,
) -> dict:
    def torch_fwd():
        conv(x)

    def torch_fwd_bwd():
        conv.zero_grad(set_to_none=True)
        y = conv(x)
        y.backward(torch.ones_like(y))

    def triton_fwd():
        depthwise_dilated_conv1d_triton(x, conv.weight, conv.bias, dilation)

    def triton_fwd_bwd():
        conv.zero_grad(set_to_none=True)
        y = depthwise_dilated_conv1d_triton(x, conv.weight, conv.bias, dilation)
        y.backward(torch.ones_like(y))

    with torch.no_grad():
        torch_fwd_ms = _time_ms(torch_fwd, num_warmup, num_iters)
        triton_fwd_ms = _time_ms(triton_fwd, num_warmup, num_iters)
    torch_fwd_bwd_ms = _time_ms(torch_fwd_bwd, num_warmup, num_iters)
    triton_fwd_bwd_ms = _time_ms(triton_fwd_bwd, num_warmup, num_iters)

    return {
        "dilation": dilation,
        "torch_fwd_ms": torch_fwd_ms,
        "triton_fwd_ms": triton_fwd_ms,
        "fwd_speedup": torch_fwd_ms / triton_fwd_ms,
        "torch_fwd_bwd_ms": torch_fwd_bwd_ms,
        "triton_fwd_bwd_ms": triton_fwd_bwd_ms,
        "fwd_bwd_speedup": torch_fwd_bwd_ms / triton_fwd_bwd_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time the Triton depthwise-conv kernel against nn.Conv1d on GPU."
    )
    parser.add_argument("--model_dim", type=int, default=1024, help="Channel width (default: %(default)s)")
    parser.add_argument("--kernel_size", type=int, default=5, help="Conv kernel size (default: %(default)s)")
    parser.add_argument("--seq_len", type=int, default=20480, help="Sequence length (default: %(default)s)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: %(default)s)")
    parser.add_argument("--num_blocks", type=int, default=24, help="Blocks to derive the dilation schedule from (default: %(default)s)")
    parser.add_argument("--num_warmup", type=int, default=10, help="Warmup iterations per timing (default: %(default)s)")
    parser.add_argument("--num_iters", type=int, default=50, help="Timed iterations per timing (default: %(default)s)")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA GPU.")
    device = torch.device("cuda")
    if not triton_dw_conv_supported(device.type, device.index):
        raise SystemExit(
            "Triton depthwise-conv kernel isn't supported here: requires triton "
            "installed and a CUDA device with compute capability >= 7.0 (Volta+)."
        )

    dtype = getattr(torch, args.dtype)
    dilations = sorted({_dilation_rate(i) for i in range(args.num_blocks)})

    print(
        f"model_dim={args.model_dim} kernel_size={args.kernel_size} "
        f"seq_len={args.seq_len} batch_size={args.batch_size} dtype={args.dtype} "
        f"gpu={torch.cuda.get_device_name(device)}"
    )
    header = (
        f"{'dilation':>8} {'torch_fwd_ms':>13} {'triton_fwd_ms':>14} {'fwd_speedup':>12} "
        f"{'torch_fb_ms':>12} {'triton_fb_ms':>13} {'fb_speedup':>11}"
    )
    print(header)
    print("-" * len(header))

    for dilation in dilations:
        conv = torch.nn.Conv1d(
            args.model_dim,
            args.model_dim,
            args.kernel_size,
            dilation=dilation,
            padding="same",
            groups=args.model_dim,
        ).to(device=device, dtype=dtype)
        x = torch.randn(
            args.batch_size, args.model_dim, args.seq_len, device=device, dtype=dtype, requires_grad=True
        )
        result = _benchmark_one(conv, x, dilation, args.num_warmup, args.num_iters)
        print(
            f"{result['dilation']:>8} {result['torch_fwd_ms']:>13.3f} {result['triton_fwd_ms']:>14.3f} "
            f"{result['fwd_speedup']:>11.2f}x {result['torch_fwd_bwd_ms']:>12.3f} "
            f"{result['triton_fwd_bwd_ms']:>13.3f} {result['fwd_bwd_speedup']:>10.2f}x"
        )


if __name__ == "__main__":
    main()
