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
| `interpret` | Run DeepLIFT/SHAP interpretation with tangermeme       |

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

The architecture uses named `nn.ReLU()` module instances (one per non-linearity) so it is compatible with [`tangermeme`](https://github.com/jmschrei/tangermeme)'s `deep_lift_shap`. Wrap the model to transpose the channels-first input expected by tangermeme and reduce the output to `(batch, 1)` (we average over positions and tracks in the demo script):

```python
import torch
import torch.nn as nn
from tangermeme.deep_lift_shap import deep_lift_shap
from promoterai_torch.utils import load_pretrained

model, args = load_pretrained("model.pt")
model.eval()

class PromoterAIWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):                           # x: (B, 4, L) channels-first
        out = self.model(x.transpose(1, 2))         # PromoterAI expects (B, L, 4)
        out = out[0].mean(dim=(1, 2)).unsqueeze(1)  # (B, 1) — mean over positions and tracks
        return out

wrapper = PromoterAIWrapper(model)

# x: (B, 4, input_length) one-hot, channels-first
x = torch.zeros(1, 4, args["input_length"])
x[0, 0, :] = 1.0  # replace with your sequences

attributions = deep_lift_shap(wrapper, x, n_shuffles=20, device="cuda", batch_size=1)
# attributions: (B, 4, input_length) — per-position, per-base importance
```

![SFSWAP DeepLIFTSHAP](https://raw.githubusercontent.com/genomicsxai/promoterai-torch/main/examples/img/deepliftshap.png)

Do note that calculating DeepLIFT/SHAP on this model is quite expensive: with TF32, `n_shuffles=20`, and `batch_size=1`, it takes ~92s/sequence with ~71GB VRAM used on an A100 80GB.

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
