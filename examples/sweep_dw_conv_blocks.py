"""
Sweep the Triton dw_conv kernel's (block_c, block_l) launch-grid block sizes.

benchmark_dw_conv.py showed the kernel roughly at parity with nn.Conv1d at low
dilation but consistently behind on the backward pass, with block_c=128/
block_l=64 -- values that were sized just enough to avoid the register-pressure
blowup fixed earlier (see notes/implementation.md), not tuned for throughput.
This sweeps a grid of alternatives at a few representative dilations (low,
mid, high -- since the taps-spread-out-in-memory effect at high dilation might
favor a different block size than low dilation) and reports the best combo per
dilation, to find a real improvement empirically rather than guessing one
value at a time.

Requires a CUDA GPU with triton installed (compute capability >= 7.0).

Example:
    python examples/sweep_dw_conv_blocks.py --model_dim 1024 --seq_len 20480 --batch_size 8
"""

from __future__ import annotations

import argparse
import itertools
import time

import torch

from promoterai_torch.triton_ops import (
    depthwise_dilated_conv1d_triton,
    triton_dw_conv_supported,
)

_BLOCK_C_CANDIDATES = (64, 128, 256, 512)
_BLOCK_L_CANDIDATES = (32, 64, 128, 256)
_DILATION_LABELS = {"low": 4, "mid": 64, "high": 1024}


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


def _sweep_one_dilation(
    conv: torch.nn.Conv1d,
    x: torch.Tensor,
    dilation: int,
    num_warmup: int,
    num_iters: int,
) -> list[dict]:
    results = []
    for block_c, block_l in itertools.product(_BLOCK_C_CANDIDATES, _BLOCK_L_CANDIDATES):

        def triton_fwd(block_c=block_c, block_l=block_l):
            depthwise_dilated_conv1d_triton(
                x, conv.weight, conv.bias, dilation, block_c=block_c, block_l=block_l
            )

        def triton_fwd_bwd(block_c=block_c, block_l=block_l):
            conv.zero_grad(set_to_none=True)
            y = depthwise_dilated_conv1d_triton(
                x, conv.weight, conv.bias, dilation, block_c=block_c, block_l=block_l
            )
            y.backward(torch.ones_like(y))

        try:
            with torch.no_grad():
                fwd_ms = _time_ms(triton_fwd, num_warmup, num_iters)
            fwd_bwd_ms = _time_ms(triton_fwd_bwd, num_warmup, num_iters)
        except Exception as exc:  # noqa: BLE001 - report and keep sweeping other combos
            results.append(
                {"block_c": block_c, "block_l": block_l, "error": str(exc)}
            )
            continue
        results.append(
            {
                "block_c": block_c,
                "block_l": block_l,
                "fwd_ms": fwd_ms,
                "fwd_bwd_ms": fwd_bwd_ms,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep block_c/block_l for the Triton dw_conv kernel."
    )
    parser.add_argument("--model_dim", type=int, default=1024, help="Channel width (default: %(default)s)")
    parser.add_argument("--kernel_size", type=int, default=5, help="Conv kernel size (default: %(default)s)")
    parser.add_argument("--seq_len", type=int, default=20480, help="Sequence length (default: %(default)s)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: %(default)s)")
    parser.add_argument("--num_warmup", type=int, default=5, help="Warmup iterations per timing (default: %(default)s)")
    parser.add_argument("--num_iters", type=int, default=20, help="Timed iterations per timing (default: %(default)s)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This sweep requires a CUDA GPU.")
    device = torch.device("cuda")
    if not triton_dw_conv_supported(device.type, device.index):
        raise SystemExit(
            "Triton depthwise-conv kernel isn't supported here: requires triton "
            "installed and a CUDA device with compute capability >= 7.0 (Volta+)."
        )

    print(
        f"model_dim={args.model_dim} kernel_size={args.kernel_size} "
        f"seq_len={args.seq_len} batch_size={args.batch_size} "
        f"gpu={torch.cuda.get_device_name(device)}"
    )

    for label, dilation in _DILATION_LABELS.items():
        conv = torch.nn.Conv1d(
            args.model_dim,
            args.model_dim,
            args.kernel_size,
            dilation=dilation,
            padding="same",
            groups=args.model_dim,
        ).to(device)
        x = torch.randn(
            args.batch_size, args.model_dim, args.seq_len, device=device, requires_grad=True
        )
        results = _sweep_one_dilation(conv, x, dilation, args.num_warmup, args.num_iters)

        print(f"\n--- dilation={dilation} ({label}) ---")
        header = f"{'block_c':>8} {'block_l':>8} {'fwd_ms':>10} {'fwd_bwd_ms':>12}"
        print(header)
        print("-" * len(header))
        ok = [r for r in results if "error" not in r]
        for r in sorted(ok, key=lambda r: r["fwd_bwd_ms"]):
            print(f"{r['block_c']:>8} {r['block_l']:>8} {r['fwd_ms']:>10.3f} {r['fwd_bwd_ms']:>12.3f}")
        for r in results:
            if "error" in r:
                print(f"{r['block_c']:>8} {r['block_l']:>8} FAILED: {r['error']}")
        if ok:
            best = min(ok, key=lambda r: r["fwd_bwd_ms"])
            print(f"best: block_c={best['block_c']} block_l={best['block_l']}")


if __name__ == "__main__":
    main()
