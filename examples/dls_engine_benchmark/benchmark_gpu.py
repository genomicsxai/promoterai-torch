"""GPU benchmark: tangermeme deep_lift_shap vs the prototype engine.

Random-weight PromoterAI at the published depth and length (24 blocks, L=20480,
4096-bp output, 498 tracks) and reduced widths, one sequence x 20 shuffles. For
each width, tangermeme runs at increasing batch sizes and the engine at increasing
ref_chunk until either runs out of memory; reports time, peak CUDA memory, and the
attributions' relative L2 difference.

    CUDA_VISIBLE_DEVICES=1 python examples/dls_engine_benchmark/benchmark_gpu.py --widths 64,128,256
"""
import argparse, time, gc, torch
from torch import nn
from tangermeme.deep_lift_shap import deep_lift_shap as tm_dls
from promoterai_torch._deep_lift_shap_engine import deep_lift_shap as engine_dls
from promoterai_torch.architecture import PromoterAI

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--widths", default="64,128,256", help="comma-separated model_dim values")
cli = parser.parse_args()
dev = torch.device("cuda:0")
L, K = 20480, 20
widths = [int(w) for w in cli.widths.split(",")]

class W(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m(x.transpose(1, 2))[0].mean(dim=(1, 2)).unsqueeze(1)

def run(fn):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    try:
        out = fn()
    except torch.OutOfMemoryError:
        return None, None, None
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**30

torch.manual_seed(0)
X = torch.eye(4)[torch.randint(0, 4, (1, L))].transpose(1, 2).contiguous()
print(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB | 24 blocks, L={L}, 1 sequence x {K} shuffles", flush=True)
for C in widths:
    torch.manual_seed(1)
    model = PromoterAI(num_blocks=24, model_dim=C, output_dims=[498], output_crop=L - 4096).to(dev).eval()
    wrapper = W(model).eval()
    run(lambda: engine_dls(model, X, n_shuffles=2, random_state=0, ref_chunk=2, device=dev))  # warm-up (cuDNN, allocator)
    best_tm, best_en = None, None
    for bs in (1, 2, 4, 10, 20):
        a, t, mem = run(lambda: tm_dls(wrapper, X, n_shuffles=K, random_state=0, batch_size=bs, device=dev, warning_threshold=1e9))
        print(f"  C={C:4d} tangermeme batch_size={bs:2d}: " + ("OOM" if t is None else f"{t:6.1f} s  peak {mem:5.2f} GiB"), flush=True)
        if t is not None and (best_tm is None or t < best_tm[1]): best_tm = (a, t, mem, bs)
        if t is None: break
    for rc in (1, 2, 4, 10, 20):
        a, t, mem = run(lambda: engine_dls(model, X, n_shuffles=K, random_state=0, ref_chunk=rc, device=dev))
        print(f"  C={C:4d} engine    ref_chunk={rc:2d}: " + ("OOM" if t is None else f"{t:6.1f} s  peak {mem:5.2f} GiB"), flush=True)
        if t is not None and (best_en is None or t < best_en[1]): best_en = (a, t, mem, rc)
        if t is None: break
    if best_tm and best_en:
        rel = float((best_en[0] - best_tm[0]).norm() / best_tm[0].norm())
        print(f"  C={C:4d} SUMMARY best tangermeme {best_tm[1]:.1f} s (bs={best_tm[3]}, {best_tm[2]:.2f} GiB) vs engine {best_en[1]:.1f} s "
              f"(chunk={best_en[3]}, {best_en[2]:.2f} GiB): {best_tm[1] / best_en[1]:.1f}x faster | attributions rel L2 {rel:.1e}", flush=True)
    elif best_en:
        print(f"  C={C:4d} SUMMARY tangermeme OOM at every batch size; engine {best_en[1]:.1f} s ({best_en[2]:.2f} GiB, chunk={best_en[3]})", flush=True)
    del model, wrapper; gc.collect(); torch.cuda.empty_cache()
