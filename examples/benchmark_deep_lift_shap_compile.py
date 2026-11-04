"""Eager vs compile=True for promoterai_torch.attribution.deep_lift_shap on one GPU.

Real converted checkpoint, one 20,480-bp promoter window (SFSWAP by default):

    uv run --extra interpret python examples/benchmark_deep_lift_shap_compile.py \
        --checkpoint /path/to/promoterAI_v1_hg38.pt --fasta /path/to/hg38.fa

On CUDA capability >= 7.0 compile=True is torch.compile (Triton); on older GPUs
it is TorchScript's fuser for the rescale rule.

Prints, per mode: first-call time (includes compilation), steady time for
1 sequence x 20 shuffles, peak CUDA memory, the auto plan, and agreement with eager.
"""
import argparse, gc, time, warnings
import pyfaidx, torch
from promoterai_torch.attribution import _plan, deep_lift_shap
from promoterai_torch.dataset import onehot_encode
from promoterai_torch.utils import load_pretrained

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--fasta", required=True)
p.add_argument("--chrom", default="chr12")
p.add_argument("--center", type=int, default=131711089)
p.add_argument("--n_shuffles", type=int, default=20)
p.add_argument("--repeats", type=int, default=2)
a = p.parse_args()
dev = torch.device("cuda")
print(f"{torch.cuda.get_device_name(dev)} (cc {torch.cuda.get_device_capability(dev)}), torch {torch.__version__}, "
      f"TF32 matmul {torch.backends.cuda.matmul.allow_tf32}", flush=True)
L = 20480
seq = str(pyfaidx.Fasta(a.fasta)[a.chrom][a.center - L // 2: a.center + L // 2]).upper()
X = torch.from_numpy(onehot_encode(seq)).float().T.unsqueeze(0).contiguous()
model, _ = load_pretrained(a.checkpoint)
model = model.to(dev).eval()

def run(**kw):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        out = deep_lift_shap(model, X, n_shuffles=a.n_shuffles, random_state=0, return_deltas=True, device=dev, **kw)
    torch.cuda.synchronize()
    msgs = sorted({str(w.message)[:120] for w in ws if "TorchScript" not in str(w.message)})
    return out, time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**30, msgs

ref = None
for comp in (False, True):
    print(f"compile={comp}: auto plan {_plan(model, L, 1, a.n_shuffles, torch.float32, dev, compile=comp)}", flush=True)
    for rc in (1, None):
        for r in range(a.repeats):
            (attr, d), t, mem, msgs = run(compile=comp, ref_chunk=rc)
            if ref is None:
                ref = attr
            rel = float((attr - ref).norm() / ref.norm())
            print(f"  compile={comp} ref_chunk={rc or 'auto'} run {r}: {t:6.1f} s  peak {mem:5.2f} GiB  "
                  f"max|delta| {float(d.max()):.1e}  rel L2 vs eager {rel:.1e}" + (f"  warnings: {msgs}" if msgs else ""), flush=True)
