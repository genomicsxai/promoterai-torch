# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Project

PyTorch port of PromoterAI — a 1D MetaFormer model that predicts the regulatory impact of promoter DNA variants on gene expression (*Science* 2025). Primary use cases: variant scoring and embedding extraction with a pretrained model, and fine-tuning on custom variant data. Training from scratch is also supported. The original TF/Keras implementation is in `PromoterAI/` (read-only reference, not published with this package).

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

Real-model equivalence examples require the licensed Illumina TF/Keras
SavedModels, converted `.pt` checkpoints, a reference FASTA, and usually CUDA:

```sh
bash examples/test_variant_scoring.sh   # variant score parity workflow
bash examples/test_tracks.sh            # full predicted-track parity workflow
```

`examples/test_tracks.sh` compares both random one-hot sequences and selected
promoters. The track parity scripts default to `--separate_loops`, which runs
the TensorFlow CUDA pass in a child process, writes temporary memmaps, exits TF
to release VRAM, then runs PyTorch. Use `--interleaved` only for debugging small
jobs where both runtimes can coexist in GPU memory.

## Package Structure

| File                                            | Purpose                                                                               |
| ----------------------------------------------- | ------------------------------------------------------------------------------------- |
| `src/promoterai_torch/architecture.py`          | `MetaFormerBlock`, `PromoterAI`, `TwinModel`                                          |
| `src/promoterai_torch/dataset.py`               | `onehot_encode`, `_prepare_sample`, `SequenceDataset`, `VariantDataset`               |
| `src/promoterai_torch/preprocess.py`            | TSS + FASTA + BigWig → HDF5                                                           |
| `src/promoterai_torch/train.py`                 | DDP training loop                                                                     |
| `src/promoterai_torch/finetune.py`              | GTEx fine-tuning via `TwinModel`                                                      |
| `src/promoterai_torch/score.py`                 | Variant effect scoring                                                                |
| `src/promoterai_torch/cli.py`                   | Unified `promoterai-torch` entry point                                                |
| `src/promoterai_torch/utils.py`                 | LR/WD scheduler, checkpoint, TF weight converter                                      |
| `examples/compare_tf_torch_tracks_random.py`    | Full-track TF/Torch parity on deterministic random one-hot sequences                  |
| `examples/compare_tf_torch_tracks_promoters.py` | Full-track TF/Torch parity on chosen genomic promoters                                |
| `examples/track_parity_utils.py`                | Shared parity helpers; handles multi-species TF dict outputs and VRAM cleanup helpers |

## Dependencies

Core install (`pip install promoterai-torch`): `torch`, `numpy`, `pandas`, `pyfaidx`, `tqdm`. No TensorFlow dependency for inference.

`[convert]` extra adds: `tensorflow`, `tf-keras`. Required for `promoterai-torch convert`; downstream users must convert the Illumina SavedModel themselves (license restrictions prevent distributing pre-converted checkpoints). Don't add a platform-conditional `tensorflow-cpu` variant here — `tf-keras` itself unconditionally requires plain `tensorflow`, so installing `tensorflow-cpu` alongside it pulls in both packages, which install conflicting builds of the same files (e.g. `tensorflow/libtensorflow_cc.so.2`) into the same site-packages path, producing an ABI-mismatched "undefined symbol" import error on Linux.

`[train]` extra adds: `h5py` (HDF5 training data), `scipy` (shift augmentation), `pybigtools` (preprocessing). Required for `preprocess`, `train`, and `SequenceDataset`.

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

- Optimizer: `AdamW`; gradient clip `max_norm=1e-4` (fine-tuning: `1.0`). Keras' `AdamW` hardcodes a non-adaptive epsilon placement that PyTorch's doesn't replicate, causing a transient (first ~1,000 steps) update-magnitude difference even with matching `epsilon=1e-7` — see "AdamW epsilon convention differs between Keras and PyTorch" in `notes/implementation.md`
- Multi-species `compute_loss` must keep an inactive species' (dummy-target, zero-weighted) loss term in the graph rather than skipping it — a hard skip leaves that head's gradient `None` instead of a real zero, so PyTorch's optimizer would then also skip its weight decay that batch, unlike Keras' `AdamW` (which applies weight decay unconditionally to every variable passed to `apply_gradients`) — see "Multi-species loss must use a soft zero, not a hard skip" in `notes/implementation.md`
- Multi-species training batches must be species-homogeneous — `build_weighted_dataloader`'s `_SpeciesBatchSampler` picks one species per whole batch (weighted by dataset size), matching Illumina's `sample_from_datasets` over already-batched per-species streams; sampling individual rows across species (the old approach) crashes `collate_fn` once species have different real output shapes — see "Multi-species batches must be species-homogeneous" in `notes/implementation.md`
- Preprocessing keeps a too-close-to-a-chromosome-edge TSS as an all-zero row (zero-weighted downstream via `x.max()==0`), matching Illumina's `generator.py`, rather than dropping it — dropping it shrinks that chromosome's sample count and skews multi-species sampling ratios — see "Boundary TSS rows should stay in the dataset as zero rows" in `notes/implementation.md`
- `convert_tf_weights` saves `species_order` (the real, first-appearance order of `shortcut_{species}{N}` weights — not necessarily human-first) into the checkpoint's `args`; any cross-framework test/tool aligning a multi-species Keras output dict to the PyTorch model's `output_heads` must use that, not a human/mouse name heuristic on the dict's keys — see "Cross-framework test harness must use convert_tf_weights' real species_order" in `notes/implementation.md`
- LR schedule (`make_lr_lambda`): linear warmup 0→10% epochs, constant 10→90%, linear decay 90→100%
- Weight decay scaled by same factor via `WeightDecayScheduler` (LambdaLR does not touch WD)
- Steps per epoch: `int(sum(dataset_sizes) / 10)`; fine-tuning: 20% of data per epoch
- Training `--batch_size` is global; DDP uses `batch_size / world_size` per rank and requires divisibility
- Checkpoint saves **base model** `state_dict` (unwrapped from DDP) on val loss improvement — fine-tuning also saves base model, not the twin wrapper
- `--auto_resume` resumes training from `<checkpoint_folder>/latest_model.pt` when present; explicit `--resume_checkpoint` takes precedence
- `SyncBatchNorm.convert_sync_batchnorm` must run **before** `DistributedDataParallel`

## Numerical Equivalence

The PyTorch port was validated against the original TF/Keras implementation on TERT, SFSWAP, and DNAJC9 promoter variants for both `hg38_finetune` and `hg38_mm10_finetune` models. Scores are numerically equivalent for individual models and their average, and for comparison against the published PromoterAI scores. Analysis notebooks are in `examples/plot_TERT.ipynb`,`examples/plot_SFSWAP.ipynb`, and `examples/plot_DNAJC9.ipynb`.

Full predicted regulatory tracks are validated by `examples/compare_tf_torch_tracks_random.py` and `examples/compare_tf_torch_tracks_promoters.py`. Multi-species TF/Keras models may return dict outputs; keep head ordering stable (`human`/`hg38` before `mouse`/`mm10`) when touching `examples/track_parity_utils.py`.

When changing parity examples, add or update tests in `tests/test_track_parity_examples.py`. These tests avoid the licensed SavedModels but cover dict-output normalization, per-head error calculation, random sequence generation, promoter sequence extraction, and CLI help smoke checks.

`tests/test_tf_gradient_equivalence.py` runs one real cross-framework training step (Keras and PyTorch, identical converted weights and batch, toy 8-block/dim-16 scale) and compares loss, raw/post-clip gradients, AdamW parameter deltas, and BatchNorm running stats — it does not need the licensed SavedModels. Along with `tests/test_convert.py`, it requires the `convert` extra (`tensorflow`/`tf-keras`) and is skipped by default, including in per-PR CI. Run both locally before merging any change to `architecture.py`, `train.py`, `finetune.py`, or the weight converter in `utils.py`:

```sh
uv sync --group dev --extra convert
uv run pytest tests/test_convert.py tests/test_tf_gradient_equivalence.py -v
```

Comparisons use cosine similarity (direction) and relative L2 norm (magnitude) per tensor, aggregated with a required pass *rate* across tensors, rather than elementwise `np.testing.assert_allclose` — see `tests/gradient_comparison_utils.py`. This mirrors alphagenome-pytorch's JAX-comparison "gradient ladder" methodology and is robust to a minority of tensors landing near a ReLU/near-zero-gradient boundary where ordinary framework floating-point noise can dominate a strict elementwise check without indicating a real bug. The shared single-step runner (`tests/keras_pytorch_step.py`) is reused by both this test and the real-checkpoint test below, so both stay in sync.

`tests/test_tf_gradient_equivalence_real.py` runs the same comparison at real, full published scale (`num_blocks=24`, `model_dim=1024`) against a real Illumina Keras SavedModel — practically GPU-only. It's skipped by default (no such SavedModel ships with this repo) unless `--keras-savedmodel-path` is passed:

```sh
uv run pytest tests/test_tf_gradient_equivalence_real.py -v -s \
    --keras-savedmodel-path /path/to/promoterai_keras_model \
    --device cuda --gradient-batch-size 2
```

`--gradient-input-length`/`--gradient-output-length` (default: the published 20480/4096) and `--device` (default: cuda if available, else cpu) are also available; see `tests/conftest.py`.

A weekly scheduled workflow (`tf-equivalence.yml`) also runs the toy-scale tests, but only as a final failsafe against TF/PyTorch API drift on `main` — it is not a pre-merge gate, so don't rely on it to catch a regression in your own PR. The real-checkpoint test isn't in that workflow (no GPU runner, no licensed SavedModel in CI) — run it yourself on a GPU box before merging changes that could plausibly behave differently at full scale.

## Key Data Details

- **hg38 output tracks**: 498 (histone marks, TF ChIP-seq, ATAC-seq, RNA-seq)
- **Fine-tuning filter**: `in_cds==0`, `spliceai<0.05`, gene in outlier set (`p_under<0.01 or p_over<0.01`); train on odd chr 1–19, val on chr 21–22
- **Score output**: `np.tanh(diff.round(4))` ∈ [−1, 1]; thresholds ±0.1 / ±0.2 / ±0.5
