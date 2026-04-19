# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PyTorch port of PromoterAI — a 1D MetaFormer model that predicts the regulatory impact of promoter DNA variants on gene expression (*Science* 2025). Primary use cases: variant scoring and embedding extraction with a pretrained model, and fine-tuning on custom variant data. Training from scratch is also supported. The original TF/Keras implementation is in `PromoterAI/` (read-only reference).

## Commands

```sh
uv sync --extra dev          # install with dev deps (includes h5py, scipy for dataset tests)
uv run pytest tests/ -v      # run all tests
uv run promoterai-torch --help   # unified CLI: preprocess | train | finetune | score
```

Multi-GPU training bypasses the CLI:

```sh
torchrun --nproc_per_node=N -m promoterai_torch.train [args]
```

## Package Structure

| File | Purpose |
| --- | --- |
| `promoterai_torch/architecture.py` | `MetaFormerBlock`, `PromoterAI`, `TwinModel` |
| `promoterai_torch/dataset.py` | `onehot_encode`, `_prepare_sample`, `SequenceDataset`, `VariantDataset` |
| `promoterai_torch/preprocess.py` | TSS + FASTA + BigWig → HDF5 |
| `promoterai_torch/train.py` | DDP training loop |
| `promoterai_torch/finetune.py` | GTEx fine-tuning via `TwinModel` |
| `promoterai_torch/score.py` | Variant effect scoring |
| `promoterai_torch/cli.py` | Unified `promoterai-torch` entry point |
| `promoterai_torch/utils.py` | LR/WD scheduler, checkpoint, TF weight converter |

## Dependencies

Core install (`pip install torch-promoterai`): `torch`, `numpy`, `pandas`, `pyfaidx`, `tensorflow-cpu`. TF-cpu is required for `promoterai-torch convert`; downstream users must convert the Illumina SavedModel themselves (license restrictions prevent distributing pre-converted checkpoints).

`[finetune]` extra adds: `h5py` (HDF5 training data), `scipy` (shift augmentation), `pyBigWig` (preprocessing). Required for `preprocess`, `train`, and `SequenceDataset`.

## Model Architecture

**1D MetaFormer** on one-hot DNA `(B, L, 4)` — channels-last throughout:

1. **Stem**: `Conv1d(4 → model_dim, kernel=1, relu)` — Glorot-uniform init
2. **MetaFormer blocks** × `num_blocks` (default 24), pre-norm:
   - `BatchNorm1d → DepthwiseConv1d(kernel=5, dilation, padding='same') → residual`
   - `BatchNorm1d → Linear(model_dim → ×4, relu) → Linear(×4 → model_dim) → residual`
   - Dilation per block `i`: `max(1, 2**(i//2 - 1))` — doubles every 2 blocks (1,1,1,1,2,2,…,1024,1024 for 24 blocks)
   - FFN Linear weights: `trunc_normal(std=0.01)`; conv weights: Glorot-uniform
3. **OutputHead** per species: averages `Linear(model_dim → output_dim)` projections from shortcut layers `[num_blocks, num_blocks-4, …, 4]`, then crops to `output_length`
4. **`encode(x)`**: returns final-block output `(B, L, model_dim)` — skip output heads entirely
5. **TwinModel**: wraps base model; only `output_heads[0]` trainable; `forward(x_ref, x_alt) → mean(out_alt − out_ref)` as `(B,)` — `tanh` applied in `score.py`, not here

Published hyperparameters: `num_blocks=24`, `model_dim=1024`, `input_length=20480`, `output_length=4096`, `output_crop=16384`.

## Data Pipeline

**Training data** (HDF5, one file per chromosome chunk — requires `[finetune]`):

- `x`: `(chunk, input_len, 4)` one-hot sequences centered on TSS
- `y`: `(chunk, output_len, n_tracks)` arcsinh-transformed BigWig signal
- Augmentation in `_prepare_sample`: random shift (truncated-normal) + 25% reverse-complement strand flip; disabled at validation/inference
- Multi-species weighted sampling: `WeightedRandomSampler` with weight ∝ dataset size, matching TF's `sample_from_datasets`
- Human chr 1–20 train, 21–22 val; non-human all chroms

**Variant data** (`VariantDataset` — core only): reads FASTA; validates ref allele; splices alt; pads/truncates alt to ref length with `N`

## Training Details

- Optimizer: `AdamW`; gradient clip `max_norm=1e-4` (fine-tuning: `1.0`)
- LR schedule (`make_lr_lambda`): linear warmup 0→10% epochs, constant 10→90%, linear decay 90→100%
- Weight decay scaled by same factor via `WeightDecayScheduler` (LambdaLR does not touch WD)
- Steps per epoch: `int(sum(dataset_sizes) / 10)`; fine-tuning: 20% of data per epoch
- Checkpoint saves **base model** `state_dict` (unwrapped from DDP) on val loss improvement — fine-tuning also saves base model, not the twin wrapper
- `SyncBatchNorm.convert_sync_batchnorm` must run **before** `DistributedDataParallel`

## Key Data Details

- **hg38 output tracks**: 498 (histone marks, TF ChIP-seq, ATAC-seq, RNA-seq)
- **Fine-tuning filter**: `in_cds==0`, `spliceai<0.05`, gene in outlier set (`p_under<0.01 or p_over<0.01`); train on odd chr 1–19, val on chr 21–22
- **Score output**: `np.tanh(diff.round(4))` ∈ [−1, 1]; thresholds ±0.1 / ±0.2 / ±0.5
