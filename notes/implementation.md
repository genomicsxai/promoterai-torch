# Implementation notes

Key decisions and fixes made during the initial port and validation of promoterai-torch.

## Weight converter (`src/promoterai_torch/utils.py`)

Single-species Keras models name output head weights `shortcut{N}/kernel` (no species prefix), while multi-species models use `shortcut_{species}{N}/kernel`. The converter now handles both:

```python
pfx = f"shortcut_{species}{block_num}" if species else f"shortcut{block_num}"
```

## Boundary handling in `VariantDataset` (`src/promoterai_torch/dataset.py`)

Variants near chromosome boundaries produce sequences shorter than `input_length`. Default behavior (`boundary="pad"`) N-pads to full length; `boundary="zeros"` returns an all-zero tensor instead (matching the original TF reference behavior). N→all-zero one-hot, so both are functionally equivalent for the model.

## pyfaidx multiprocessing fix

Opening a `pyfaidx.Fasta` in the parent process then forking DataLoader workers causes concurrent fd seeks and non-deterministic ref mismatches. Fix: store `_fasta_path` string in `VariantDataset` and open lazily per worker via a property. `score.py` now passes the path string, not a pre-opened handle.

## DeepLIFT/SHAP compatibility (`src/promoterai_torch/architecture.py`)

`tangermeme.deep_lift_shap` requires every non-linearity to be a distinct `nn.Module` instance defined in `__init__` and called exactly once per forward pass. All `F.relu()` calls were replaced with named module attributes:

- `MetaFormerBlock`: single `self.act = nn.ReLU()` for the FFN
- `OutputHead`: `self.acts = nn.ModuleList([nn.ReLU() for _ in self.shortcut_indices])` — one per projection
- `PromoterAI`: `self.stem_act = nn.ReLU()` for the stem

`tangermeme` expects `(B, 4, L)` channels-first input and `(B, n_targets)` output; wrap the model to transpose and reduce:

```python
class PromoterAIWrapper(nn.Module):
    def forward(self, x):           # x: (B, 4, L)
        out = self.model(x.transpose(1, 2))
        return out[0].mean(dim=1)   # (B, n_tracks)
```

## Dependency layout

- Core (`pip install promoterai-torch`): `torch`, `numpy`, `pandas`, `pyfaidx`, `tqdm` — no TensorFlow
- `[convert]`: `tensorflow`, `tf-keras` — don't add a platform-conditional `tensorflow-cpu` here; `tf-keras` already requires plain `tensorflow` unconditionally, and installing both corrupts site-packages (see AGENTS.md's Dependencies section)
- `[train]`: `h5py`, `scipy`, `pyBigWig`
- `[dev]`: `pytest`, `h5py`, `scipy`, `tangermeme>=0.5`

Downstream users must convert the Illumina SavedModel themselves (license restrictions prevent distributing pre-converted checkpoints).

## AdamW epsilon convention differs between Keras and PyTorch

`keras.optimizers.AdamW.update_step` (`tf_keras/src/optimizers/adamw.py`) computes, using the
*biased* (pre-bias-correction) moments `m`, `v`:

```python
alpha_t = lr * sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t)
update = alpha_t * m / (sqrt(v) + epsilon)
```

`torch.optim.AdamW` (and the textbook Adam formulation) bias-corrects `m`/`v` first instead:

```python
m_hat, v_hat = m / (1 - beta_1 ** t), v / (1 - beta_2 ** t)
update = lr * m_hat / (sqrt(v_hat) + epsilon)
```

These are only mathematically equivalent if `epsilon` is rescaled by `sqrt(1 - beta_2 ** t)` —
which is exactly what `keras.optimizers.Adam`'s `adaptive_epsilon` option does, but `AdamW` has
no such option and always uses the raw `epsilon` value as-is. With a shared `epsilon=1e-7`
(matching Keras' default, per `build_train_optimizer`/`build_finetune_optimizer`), Keras' step-1
update is damped by up to ~32x more than PyTorch's (`1 / sqrt(1 - 0.999**1)`), converging to
<1% difference by step ~1000 as `beta_2 ** t -> 0`.

This is a framework-inherent AdamW quirk, not a porting bug — PyTorch's built-in `AdamW` has no
option to replicate Keras' non-adaptive epsilon placement, and the effect is confined to roughly
the first ~1,000 optimizer steps of a training run (real training runs for far longer), so it
isn't worth a custom optimizer to chase. Confirmed in `tests/test_tf_gradient_equivalence.py`,
which shows the two frameworks' AdamW *mechanics* (bias correction, decoupled weight decay)
match exactly once epsilon is isolated out (`epsilon -> 0` on both sides).

## Numerical equivalence (`examples/`)

Validated on TERT (*n*=6,006), SFSWAP, and DNAJC9 promoter variants using the `hg38_finetune` and `hg38_mm10_finetune` models. Scores are numerically identical to the original TF/Keras implementation across all comparisons (r=1.0000, MAE=0.0000), including the ensembled score against published PromoterAI output.

Analysis notebooks: `examples/promoter_ism_benchmark/plot_TERT.ipynb`, `examples/promoter_ism_benchmark/plot_SFSWAP.ipynb`, `examples/promoter_ism_benchmark/plot_DNAJC9.ipynb`.

Data files per gene: `{model}.TF.tsv.gz` (original TF scores), `{model}.TORCH.tsv.gz` (PyTorch scores), `{gene}.scores.tsv.gz` (combined, cached).

## PyPI publishing (`.github/workflows/publish.yml`)

Uses PyPI trusted publishing (OIDC) — no API token required. Triggered on GitHub release publication. Environment named `pypi` with `id-token: write` permission.
