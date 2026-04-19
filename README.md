# promoterai-torch

A PyTorch port of [PromoterAI](https://github.com/Illumina/PromoterAI) v1 from Illumina — a deep learning model published in *Science* (2025) that predicts the regulatory impact of promoter DNA variants on gene expression. Note that this is **not** an official Illumina product or publication; I have no affiliation with Illumina and the original authors were not informed of this port prior to its release.

## Install

For running inference, converting pretrained TensorFlow models, or loading PyTorch models:

```sh
pip install torch-promoterai
# or with uv:
uv add torch-promoterai
```

For fine-tuning or training from scratch, install the additional dependencies:

```sh
pip install "torch-promoterai[train]"
# or with uv:
uv add "torch-promoterai[finetune]"
```

## Convert a pretrained Keras model

Download the pretrained PromoterAI SavedModel from [Illumina/PromoterAI](https://github.com/Illumina/PromoterAI), then convert it to a PyTorch checkpoint:

```sh
promoterai-torch convert \
    --keras_model models/promoterAI_v1_hg38_mm10_finetune \
    --output models/promoterAI_v1_hg38_mm10_finetune.pt \
    --input_length 20480 \
    --output_length 4096
```

Architecture parameters (`num_blocks`, `model_dim`, `output_dims`) are inferred automatically from the Keras model. `--input_length` and `--output_length` are optional metadata.

## Quick start

### Score variants

Given a pretrained checkpoint and a variant TSV with columns `chrom`, `pos`, `ref`, `alt`, `strand`:

```sh
promoterai-torch score \
    --model_checkpoint models/promoterAI_v1_hg38_mm10_finetune.pt \
    --var_file variants.tsv \
    --fasta_file hg38.fa \
    --input_length 20480
```

Scores are written to `variants.best_model.tsv` as a new `score` column in [−1, 1]. Thresholds: ±0.1 (weak effect), ±0.2 (moderate), ±0.5 (strong).

### Run inference on a genomic sequence

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

The output is one tensor per species head (only one head in the published model). Each tensor has shape `(batch, output_length, n_tracks)` where `n_tracks=498` for the published hg38 model (histone marks, TF ChIP-seq, ATAC-seq, RNA-seq).

### Extract embeddings

```python
import torch
from promoterai_torch.dataset import onehot_encode
from promoterai_torch.utils import load_pretrained

model, args = load_pretrained("best_model.pt")
model.eval()

seq = "ACGT" * (args["input_length"] // 4)   # replace with your sequence
x = torch.from_numpy(onehot_encode(seq)).unsqueeze(0)

with torch.no_grad():
    embeddings = model.encode(x)   # (1, input_length, model_dim)
```

`model.encode()` returns the final MetaFormer block output — a per-position representation of shape `(B, L, model_dim)` suitable for downstream tasks.

## Training from scratch

**WARNING** This functionality is completely untested. It was just auto-ported by Claude because this function was present in the original PromoterAI repo. I'm keeping it here in case someone might find it useful, but I haven't done any work to verify that this is correct or even runs.

Preprocess one chromosome at a time (parallelizable), then train:

```sh
promoterai-torch preprocess \
    --hdf5_folder data/hdf5/human --tss_file tss_hg38.tsv \
    --fasta_file hg38.fa --bigwig_files hg38_tracks.tsv \
    --chrom chr1 --input_length 32768 --output_length 16384

promoterai-torch train \
    --checkpoint_folder checkpoints/run1 \
    --hdf5_human_folder data/hdf5/human \
    --input_length 20480 --output_length 4096 \
    --num_blocks 24 --model_dim 1024 --batch_size 32
```

Multi-GPU via `torchrun`:

```sh
torchrun --nproc_per_node=4 -m promoterai_torch.train \
    --checkpoint_folder checkpoints/run1 \
    --hdf5_human_folder data/hdf5/human \
    --hdf5_nonhuman_folders data/hdf5/mouse \
    --input_length 20480 --output_length 4096 \
    --num_blocks 24 --model_dim 1024 --batch_size 32
```

### Fine-tune on custom variants

```sh
promoterai-torch finetune \
    --model_checkpoint best_model.pt \
    --var_file data/annotation/finetune_gtex.tsv \
    --fasta_file hg38.fa \
    --input_length 20480 \
    --batch_size 8 \
    --epochs 100
```

The fine-tuned base model is saved to `best_model_finetune/best_model.pt`. Only the first output head is trained; all other weights are frozen.

The variant TSV must include `chrom`, `pos`, `ref`, `alt`, `strand`, `z` (expression z-score target), `in_cds`, `spliceai`, `p_under`, `p_over`, `gene` columns (matching the GTEx outlier format).

## Development

```sh
uv sync --extra dev
uv run pytest tests/ -v
```

## Reference

Jaganathan, Ersaro, Novakovsky et al. *Science* (2025).
Original TF implementation: [Illumina/PromoterAI](https://github.com/Illumina/PromoterAI)
