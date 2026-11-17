# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break
things).

## [Unreleased]

### Changed

- Document `utils.set_tf32` as the way to match TensorFlow's TF32 default from
  the Python API (library functions, including `deep_lift_shap`, leave the
  global flags alone). `examples/deepliftshap_region.py` gains
  `--tf32`/`--no-tf32` (default on).
- `--tf32` and `--incremental` no longer print their default twice in `--help`
  on Python < 3.12.

### Fixed

- Toy TF-equivalence tests no longer fail with "Visible devices cannot be
  modified after being initialized" when a GPU is visible: `conftest.py` hides
  GPUs from TensorFlow at pytest startup.

## [0.6.0] - 2026-11-10

### Changed

- `score`, `train`, `finetune` and the paper benchmark script now allow TF32
  matmuls and convolutions by default (`utils.set_tf32(True)`), matching
  TensorFlow and the original PromoterAI package; pass `--no-tf32` for strict
  float32. TF32 moves ~10% of 4-decimal scores by 1e-4.
- `benchmark.score_variants` takes `tf32=None` and leaves the global flags
  alone by default.

### Added

- Scoring speed against the TensorFlow package in
  [docs/numerical-equivalence.md](docs/numerical-equivalence.md#scoring-speed-vs-tensorflow).

## [0.5.0] - 2026-11-10

### Added

- `deep_lift_shap(target_weights=...)`: attribute any weighted sum of a head's
  output window (per-track weights, position windows, full position × track
  weights, signed contrasts). Mutually exclusive with `tracks=`.
- `examples/deepliftshap_region.py --species`, resolved through the
  checkpoint's `species_order`.

### Changed

- README trimmed to a quickstart; installation and DeepLIFT/SHAP details moved
  to [docs/installation.md](docs/installation.md) and
  [docs/attribution.md](docs/attribution.md).
- Documented that `head=` follows the checkpoint's `species_order`, which is
  not guaranteed to be human-first.

## [0.4.0] - 2026-11-08

### Added

- `promoterai_torch.attribution.deep_lift_shap`: a memory-lean DeepLIFT/SHAP
  engine that reproduces `tangermeme.deep_lift_shap` (to ~1e-16 in float64),
  with automatic batch/reference chunk planning, out-of-memory retry, `tracks=`
  selection and `compile=True` (torch >= 2.6). New `interpret` extra
  (tangermeme, for shuffled references and logo plots).
- Incremental (fastISM-style) scoring of substitutions in `score` and the paper
  benchmark, exact up to rounding and the new default (`--no-incremental` to
  disable); ~5× faster on TERT promoter SNVs.
- uv dependency groups `cu126`, `cu130` and `cpu` to choose the PyTorch build
  (CUDA 12.6 supports Maxwell/Pascal/Volta GPUs).
- `preprocess --threads`: multithreaded BigWig reads, per-sample finishing and
  gzip compression.
- Python 3.14 support.

### Changed

- `convert` reads Keras SavedModels with numpy alone (`savedmodel.py`); no
  TensorFlow needed. **The `convert` extra is removed.**
- BigWig reading uses `figwig` instead of `pybigtools`.
- Preprocessing chunks HDF5 per sample and along the length, and training
  reads only the crop window. Files written by older versions still load.
- `convert` builds the target model on the meta device.

## [0.3.0] - 2026-08-28

### Fixed

- Multi-species training: batches are now species-homogeneous
  (`_SpeciesBatchSampler`), fixing a `collate_fn` crash when species have
  different output shapes.
- Multi-species loss keeps an inactive species' zero-weighted term in the
  graph, so weight decay applies to every head every step, as in Keras.
- Preprocessing keeps TSSs too close to a chromosome edge as zero-weighted
  all-zero rows instead of dropping them, preserving sampling ratios.
- The `convert` extra no longer installs both `tensorflow` and
  `tensorflow-cpu` on Linux (which broke `import tf_keras`).

### Added

- `convert_tf_weights` records `species_order` in the checkpoint.
- Cross-framework training-step equivalence tests against Keras (loss,
  gradients, AdamW updates, BatchNorm statistics), at toy scale and on the
  real published checkpoints, including fine-tuned ones.

## [0.2.1] - 2026-08-19

### Fixed

- `train` steps per epoch were `batch_size` times too large.
- `train` AdamW epsilon now matches Keras (1e-7).
- Gradient clipping is per parameter (Keras `clipnorm`), not one global norm.
- BatchNorm momentum now matches Keras (0.01 in PyTorch's convention); affects
  training from scratch only.

## [0.2.0] - 2026-08-18

### Changed

- Preprocessing writes one HDF5 file per chromosome instead of many chunk
  files; training raises if a chromosome file is missing.
- One-hot encoding is numba-accelerated; non-ACGT bases (including IUPAC
  codes) encode as all-zero rows instead of raising.
- `numba` is a core dependency; `scipy` is dropped.
- The `promoterai-torch` CLI and `python -m promoterai_torch.<module>` share
  one argument parser per command.
- `input_length` falls back to the checkpoint metadata, then to 20480.

### Removed

- Unused `WeightDecayScheduler`.

## [0.1.0.post1] - 2026-08-18

### Fixed

- README images and badges on PyPI.

## [0.1.0] - 2026-08-14

Initial release.

[Unreleased]: https://github.com/genomicsxai/promoterai-torch/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/genomicsxai/promoterai-torch/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/genomicsxai/promoterai-torch/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/genomicsxai/promoterai-torch/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/genomicsxai/promoterai-torch/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/genomicsxai/promoterai-torch/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/genomicsxai/promoterai-torch/compare/v0.1.0...v0.2.0
[0.1.0.post1]: https://github.com/genomicsxai/promoterai-torch/compare/v0.1.0...665d0c4
[0.1.0]: https://github.com/genomicsxai/promoterai-torch/releases/tag/v0.1.0
