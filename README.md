# promoterai-torch

[![PyPI](https://img.shields.io/pypi/v/promoterai-torch)](https://pypi.org/project/promoterai-torch/) [![Tests](https://github.com/genomicsxai/promoterai-torch/actions/workflows/tests.yml/badge.svg)](https://github.com/genomicsxai/promoterai-torch/actions/workflows/tests.yml) [![PyPI Downloads](https://static.pepy.tech/personalized-badge/promoterai-torch?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/promoterai-torch)

A PyTorch port of [PromoterAI](https://github.com/Illumina/PromoterAI) v1 from Illumina — a deep learning model that predicts the regulatory impact of promoter DNA variants on gene expression.

> [!Important]
> This is **not** an official Illumina product or publication. The contents of this package are solely the responsibility of the authors/maintainers and its release should not be construed as being supported/endorsed by Illumina or the original authors of PromoterAI.
>
> The official PromoterAI codebase is released under the [PolyForm Strict License 1.0.0](https://polyformproject.org/licenses/strict/1.0.0/). The pretrained models and precomputed variant scores are released separately under Illumina's own academic-only-use data license (see [their github](https://github.com/Illumina/PromoterAI) for instructions on academic/commercial licensing). This package is MIT-licensed and contains no Illumina code, models, or scores, but official PromoterAI models are bound by Illumina's license terms, even after being converted. Please do not redistribute converted checkpoints.

## Install

Python 3.10, 3.11, 3.12, 3.13, and 3.14 are supported.

For variant scoring, embedding extraction, and ordinary PyTorch inference from
an already-converted checkpoint, install the core package:

```sh
pip install promoterai-torch
```

Or with [uv](https://docs.astral.sh/uv/):

```sh
uv add promoterai-torch
```

The core package also converts the official Keras SavedModels (no TensorFlow
needed). Optional workflows are split into extras so inference installs do not
pull in HDF5/BigWig tooling or attribution libraries:

| Extra       | Enables                                              |
| ----------- | ----------------------------------------------------- |
| `train`     | Preprocess data, train from scratch, or fine-tune      |
| `wandb`     | Weights & Biases logging (combine with `train`)        |
| `interpret` | DeepLIFT/SHAP references and logo plots (tangermeme)   |

`uv add` is for installing into an existing project; for a
cloned checkout with development dependencies, see
[CONTRIBUTING.md](CONTRIBUTING.md).

### Choosing a PyTorch build

`pip install promoterai-torch` and `uv add promoterai-torch` take `torch` from
PyPI. On Linux that build uses CUDA 13.0 and supports NVIDIA GPUs of compute
capability 7.5 (Turing) and newer. For an older GPU, or a CPU-only machine,
install a different build from PyTorch's own package index:

| GPU | Compute capability | Build |
| --- | --- | --- |
| Maxwell, Pascal, Volta (e.g. GTX 10xx, TITAN X/Xp, P100, V100) | 5.0–7.0 | CUDA 12.6 (`cu126`) |
| Turing and newer (RTX 20xx+, T4, A100, H100, Blackwell) | 7.5+ | PyPI default, or `cu130` |
| None | — | `cpu` |

Check your GPU's compute capability with
`nvidia-smi --query-gpu=name,compute_cap --format=csv`. A build that doesn't
support your GPU fails at the first CUDA operation with
"no kernel image is available for execution on the device".

With pip, install torch from the matching index first, then the package:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or cu130, cpu
pip install promoterai-torch
```

In a clone of this repository, uv selects the build with a dependency group
(`cu126`, `cu130` or `cpu`; use at most one):

```sh
uv sync --group cu126                          # CUDA 12.6 build, for Pascal etc.
uv sync --group dev --extra train --group cu126
uv run --group cu126 pytest tests/             # pass the same group to uv run
```

`uv sync` and `uv run` without the group switch the environment back to the
PyPI build (and download it), so pass the group every time. To keep several
builds side by side, give each its own environment:

```sh
export UV_PROJECT_ENVIRONMENT=/path/to/venv-cu126
uv sync --group dev --group cu126
uv run --group cu126 promoterai-torch score ...
```

The CUDA 12.6 line is PyTorch's legacy build: its last release is older than
the newest PyPI torch, so a `cu126` environment may resolve an older torch
version than the default one.

## Convert a pretrained Keras model

Download the pretrained PromoterAI SavedModel from [Illumina/PromoterAI](https://github.com/Illumina/PromoterAI) and convert it to a PyTorch checkpoint. Conversion reads the SavedModel files directly with numpy, so it works with the core install and does not need TensorFlow:

```sh
promoterai-torch convert \
    --keras_model models/promoterAI_v1_hg38_mm10_finetune \
    --output models/promoterAI_v1_hg38_mm10_finetune.pt \
    --input_length 20480 \
    --output_length 4096
```

Architecture parameters (`num_blocks`, `model_dim`, `output_dims`) are inferred automatically from the Keras model. (The `[convert]` extra that earlier versions used for this is gone; `pip install "promoterai-torch[convert]"` still installs the package, with a warning that the extra doesn't exist.) `--input_length` and `--output_length` are optional metadata.

## Usage

### Score variants

Given a pretrained checkpoint and a variant TSV with columns `chrom`, `pos`, `ref`, `alt`, `strand`:

```sh
promoterai-torch score \
    --model_checkpoint models/promoterAI_v1_hg38_mm10_finetune.pt \
    --var_file variants.tsv \
    --fasta_file hg38.fa \
    --input_length 20480
```

Scores are written by default to `variants.{model_name}.tsv` as a new `score` column in [−1, 1] (or to a file path provided by `--output`). Thresholds: ±0.1 (weak effect), ±0.2 (moderate), ±0.5 (strong).

Same-length substitutions (SNVs and MNVs) are scored incrementally by default:
each site's reference window runs through the model once, and each alt then
recomputes only the window of every layer its edit can reach (the fastISM idea,
exact up to floating-point rounding), so the three SNVs at a site cost about one
full pass plus three small ones instead of six full passes. On 150 TERT promoter
SNVs (50 sites) this took 100 s instead of 527 s on a TITAN Xp, with identical
scores. Indels always use full ref/alt passes. Pass `--no-incremental` to score
everything with full passes. The paper benchmark
(`examples/paper_benchmark/benchmark_variant_scores.py`) scores the same way and
takes the same flag.

### Run inference on a genomic sequence

One can also generate predictions for all the tracks that PromoterAI was trained on (these are aggregated and diff'ed to generate the variant scores).

```python
import torch
from promoterai_torch.dataset import onehot_encode
from promoterai_torch.utils import load_pretrained

model, args = load_pretrained("models/promoterAI_v1_hg38_mm10_finetune.pt")
model.eval()

# One-hot encode a DNA sequence → (L, 4), add batch dim → (1, L, 4)
# Use the full input_length the model was trained on (20480 bp for the published model)
seq = "ACGT" * (args["input_length"] // 4)   # replace with your sequence
x = torch.from_numpy(onehot_encode(seq)).unsqueeze(0)

with torch.no_grad():
    predictions = model(x)   # tuple of (1, output_length, n_tracks) per output head

track_predictions = predictions[0]   # (1, output_length, n_tracks) — arcsinh-scale signal
```

The output is one tensor per species head. Each tensor has shape `(batch, output_length, n_tracks)` where `n_tracks=498` for the published human head (histone marks, TF ChIP-seq, ATAC-seq, RNA-seq) and `n_tracks=472` for the mouse head (per Illumina's released `data/bigwig/mm10.tsv` track list).

### Extract embeddings

```python
import torch
from promoterai_torch.dataset import onehot_encode
from promoterai_torch.utils import load_pretrained

model, args = load_pretrained("model.pt")
model.eval()

seq = "ACGT" * (args["input_length"] // 4)   # replace with your sequence
x = torch.from_numpy(onehot_encode(seq)).unsqueeze(0)

with torch.no_grad():
    embeddings = model.encode(x)   # (1, input_length, model_dim)
```

`model.encode()` returns the final MetaFormer block output — a per-position representation of shape `(B, L, model_dim)` suitable for downstream tasks.

### DeepLIFT/SHAP attribution

Install the optional interpretation dependencies first:

```sh
pip install "promoterai-torch[interpret]"
# or
uv add promoterai-torch --extra interpret
```

`promoterai_torch.attribution.deep_lift_shap` computes DeepLIFT/SHAP attributions of
the model's mean prediction with an engine written for this architecture. It gives
the same attributions as [`tangermeme`](https://github.com/jmschrei/tangermeme)'s
`deep_lift_shap` (same dinucleotide-shuffled references for a given `random_state`,
same rescale rule), but the full published model fits on a 10 GB GPU:

```python
from promoterai_torch.attribution import deep_lift_shap
from promoterai_torch.utils import load_pretrained

model, args = load_pretrained("model.pt")
model = model.to("cuda")

# X: (N, 4, input_length) one-hot, channels-first (tangermeme's layout)
attributions, deltas = deep_lift_shap(
    model, X,
    head=0,             # which species head
    tracks=None,        # or a list of track indices; the output is their mean
    n_shuffles=20, random_state=0,
    return_deltas=True, # |sum(attr) - (y(x) - y(ref))| per reference, ~1e-7 in float32
)
# attributions: (N, 4, input_length) on the CPU, already multiplied by X
# (hypothetical=True returns every base's attribution instead)
```

How many references (and, for very short sequences, how many sequences) go through
the GPU per step is chosen to saturate it within the free device memory (at the
published scale, up to 4 references per step: ~16 GB, ~10% faster than 1 on an A100;
12 GB cards fit 2, 10 GB cards 1); pass `ref_chunk` / `batch_size` / `max_memory` to
override it. `compile=True` fuses the
elementwise work: `torch.compile` on CPUs and CUDA GPUs of compute capability 7.0+,
and TorchScript's fuser on older GPUs (e.g. Pascal), where `torch.compile`'s Triton
backend doesn't run. Both are measured below: TorchScript on Pascal is ~12% faster,
and `torch.compile` is ~21% faster on an RTX 3080 and 12-40% faster on an A100 once
compiled (the first call adds 12-70 s of compilation), agreeing with eager to a
relative L2 of ~1e-7. `torch.compile` here needs torch >= 2.6; eager mode works from
torch 2.1.

On Ampere and newer GPUs, PyTorch runs convolutions in TF32 by default
(`torch.backends.cudnn.allow_tf32 = True`), so GPU attributions differ from
tangermeme's by a relative L2 of ~1e-4 (2.7e-4 measured on an RTX 3080). This is
floating-point noise, not an engine difference, and it matches the original
TensorFlow model, which also uses TF32 on these GPUs. For an exact comparison, set
`torch.backends.cudnn.allow_tf32 = False` first; the gap then drops to ~1e-7, at no
cost in speed (1.7e-6 at the published scale on an A100, where the default gives 3e-4).

Turning TF32 on for matmuls too (`torch.backends.cuda.matmul.allow_tf32 = True`, off
by default) makes both implementations ~3x faster on an A100, at ~2e-4 relative L2
from strict float32 on the real hg38 model (~3e-3 on random weights) and convergence
deltas ~1e-3 instead of ~1e-6: fine for logo plots, not for exact checks.

Full published model (24 blocks x 1024, 20,480 bp), one sequence x 20 shuffles,
float32 with PyTorch's default precision flags unless noted (A100 rows: strict
float32, or TF32 for both convolutions and matmuls):

| Hardware | Method | Time | Peak GPU memory |
| --- | --- | --- | --- |
| A100 80 GB | `tangermeme.deep_lift_shap`, `batch_size=1` | 44 s | 65.5 GB |
| A100 80 GB | `deep_lift_shap` | 32 s | 15.6 GB |
| A100 80 GB | `deep_lift_shap(compile=True)` | 28 s | 14.1 GB |
| A100 80 GB, TF32 | `tangermeme.deep_lift_shap`, `batch_size=1` | 14 s | 65.5 GB |
| A100 80 GB, TF32 | `deep_lift_shap` | 9.9 s | 15.6 GB |
| A100 80 GB, TF32 | `deep_lift_shap(compile=True)` | 5.9 s | 14.1 GB |
| TITAN X (Pascal) 12 GB | `tangermeme.deep_lift_shap` | out of memory | |
| TITAN X (Pascal) 12 GB | `deep_lift_shap` | 96 s | 6.5 GB |
| TITAN X (Pascal) 12 GB | `deep_lift_shap(compile=True)` | 86 s | 7.7 GB |
| TITAN Xp 12 GB | `deep_lift_shap` | 78 s | 6.5 GB |
| TITAN Xp 12 GB | `deep_lift_shap(compile=True)` | 70 s | 7.7 GB |
| RTX 3080 10 GB | `tangermeme.deep_lift_shap` | out of memory | |
| RTX 3080 10 GB | `deep_lift_shap` | 40 s | 6.5 GB |
| RTX 3080 10 GB | `deep_lift_shap(compile=True)` | 32 s | 5.9 GB |

`examples/deepliftshap_region.py` attributes a genomic region and plots it;
`examples/benchmark_deep_lift_shap.py` compares the two implementations at reduced
width; `examples/benchmark_deep_lift_shap_compile.py` times eager vs `compile=True`
on a real checkpoint.

![SFSWAP DeepLIFTSHAP](https://raw.githubusercontent.com/genomicsxai/promoterai-torch/main/examples/img/deepliftshap.png)

The engine relies on PromoterAI's structure (ReLU as the only nonlinearity,
eval-mode BatchNorm, row-wise ops). For other attribution targets or methods, the
model still works with tangermeme directly: its non-linearities are distinct
`nn.ReLU()` modules, as tangermeme requires. Wrap it to take `(B, 4, L)` and return
`(B, n_targets)`, e.g. `model(x.transpose(1, 2))[0].mean(dim=(1, 2)).unsqueeze(1)`.

## Numerical equivalence

This port produces near-identical scores and regulatory track predictions to
the original TensorFlow/Keras implementation, matching the published AUROCs.
See [docs/numerical-equivalence.md](docs/numerical-equivalence.md) for
benchmark reproduction steps, per-variant concordance results, and
full-track comparison scripts.

![SFSWAP scatter](https://raw.githubusercontent.com/genomicsxai/promoterai-torch/main/examples/img/paper_benchmark_concordance.png)

## Training models

Fine-tuning or training from scratch using the built-in scripts requires the
`train` extra described in [Install](#install) (with an optional `wandb` extra
for wandb.ai integration. See [docs/training.md](docs/training.md) for data
preprocessing, training from scratch, fine-tuning on variants, and multi-GPU usage.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setting up a local development
environment and running the test suite.

## Reference

Jaganathan, Ersaro, Novakovsky et al. *Science* (2025) Predicting expression-altering promoter mutations with deep learning. doi:10.1126/science.ads7373

He, Kundaje. *Genomics x AI Blog* (2026) [promoterai-torch: a PyTorch port of Illumina's PromoterAI](https://genomicsxai.github.io/blogs/2026-014/). doi:10.5281/zenodo.23038831

Original TF implementation: [Illumina/PromoterAI](https://github.com/Illumina/PromoterAI)

Citation metadata for this software is available in [CITATION.cff](CITATION.cff).
