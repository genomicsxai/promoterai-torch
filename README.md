# promoterai-torch

[![PyPI](https://img.shields.io/pypi/v/promoterai-torch)](https://pypi.org/project/promoterai-torch/) [![Tests](https://github.com/genomicsxai/promoterai-torch/actions/workflows/tests.yml/badge.svg)](https://github.com/genomicsxai/promoterai-torch/actions/workflows/tests.yml) [![PyPI Downloads](https://static.pepy.tech/personalized-badge/promoterai-torch?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/promoterai-torch)

A PyTorch port of [PromoterAI](https://github.com/Illumina/PromoterAI) v1 from Illumina — a deep learning model that predicts the regulatory impact of promoter DNA variants on gene expression.

> [!Important]
> This is **not** an official Illumina product or publication. The contents of this package are solely the responsibility of the authors/maintainers and its release should not be construed as being supported/endorsed by Illumina or the original authors of PromoterAI.
>
> The official PromoterAI codebase, models, and variant scores are released under fairly restrictive licensing (see [their github](https://github.com/Illumina/PromoterAI) for instructions on academic/commercial licensing). Please do not redistribute converted checkpoints.

## Install

Python 3.10, 3.11, 3.12, and 3.13 are supported.

For variant scoring, embedding extraction, and ordinary PyTorch inference from
an already-converted checkpoint, install the core package:

```sh
pip install promoterai-torch
```

Or with [uv](https://docs.astral.sh/uv/):

```sh
uv add promoterai-torch
```

Optional workflows are split into extras so inference installs do not pull in
TensorFlow, HDF5/BigWig tooling, or attribution libraries:

| Extra       | Enables                                              |
| ----------- | ----------------------------------------------------- |
| `convert`   | Convert Keras/TensorFlow SavedModels to PyTorch checkpoints |
| `train`     | Preprocess data, train from scratch, or fine-tune      |
| `wandb`     | Weights & Biases logging (combine with `train`)        |
| `interpret` | Run DeepLIFT/SHAP interpretation with tangermeme       |

`uv add` is for installing into an existing project; for a
cloned checkout with development dependencies, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Convert a pretrained Keras model

First install the `[convert]` extra (see above), then download the pretrained PromoterAI SavedModel from [Illumina/PromoterAI](https://github.com/Illumina/PromoterAI) and convert it to a PyTorch checkpoint:

```sh
pip install "promoterai-torch[convert]"
# or 
uv add promoterai-torch --extra convert
```

```sh
promoterai-torch convert \
    --keras_model models/promoterAI_v1_hg38_mm10_finetune \
    --output models/promoterAI_v1_hg38_mm10_finetune.pt \
    --input_length 20480 \
    --output_length 4096
```

Architecture parameters (`num_blocks`, `model_dim`, `output_dims`) are inferred automatically from the Keras model. `--input_length` and `--output_length` are optional metadata.

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

The output is one tensor per species head. Each tensor has shape `(batch, output_length, n_tracks)` where `n_tracks=498` for the published human head (histone marks, TF ChIP-seq, ATAC-seq, RNA-seq) and `n_tracks=??` for the mouse head.

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

Original TF implementation: [Illumina/PromoterAI](https://github.com/Illumina/PromoterAI)

Citation metadata for this software is available in [CITATION.cff](CITATION.cff).
