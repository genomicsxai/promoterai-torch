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

`promoterai_torch.attribution.deep_lift_shap` computes DeepLIFT/SHAP attributions of
the model's mean prediction with an engine written for this architecture. It gives
the same attributions as [`tangermeme`](https://github.com/jmschrei/tangermeme)'s
`deep_lift_shap` (same dinucleotide-shuffled references for a given `random_state`,
same rescale rule), but the full published model fits on a 12 GB GPU:

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
the GPU per step is chosen to saturate it within the free device memory; pass
`ref_chunk` / `batch_size` / `max_memory` to override it. `compile=True` fuses the
elementwise work: `torch.compile` on CPUs and CUDA GPUs of compute capability 7.0+,
and TorchScript's fuser on older GPUs (e.g. Pascal), where `torch.compile`'s Triton
backend doesn't run. The Pascal path is measured below (~12% faster); the
`torch.compile` GPU path is so far only validated for correctness on CPU, not
benchmarked on a 7.0+ GPU.

Full published model (24 blocks x 1024, 20,480 bp), one sequence x 20 shuffles,
float32:

| Hardware | Method | Time | Peak GPU memory |
| --- | --- | --- | --- |
| A100 80 GB, TF32 | `tangermeme.deep_lift_shap`, `batch_size=1` | ~92 s | ~71 GB |
| TITAN X (Pascal) 12 GB | `tangermeme.deep_lift_shap` | out of memory | |
| TITAN X (Pascal) 12 GB | `deep_lift_shap` | 96 s | 6.5 GB |
| TITAN X (Pascal) 12 GB | `deep_lift_shap(compile=True)` | 86 s | 7.7 GB |
| TITAN Xp 12 GB | `deep_lift_shap` | 78 s | 6.5 GB |
| TITAN Xp 12 GB | `deep_lift_shap(compile=True)` | 70 s | 7.7 GB |

`examples/deepliftshap_region.py` attributes a genomic region and plots it;
`examples/benchmark_deep_lift_shap.py` compares the two implementations at reduced
width.

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
