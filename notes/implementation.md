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
- `[convert]`: `tensorflow-cpu` (Linux/Windows) or `tensorflow` (macOS), `tf-keras`
- `[train]`: `h5py`, `scipy`, `pyBigWig`
- `[dev]`: `pytest`, `h5py`, `scipy`, `tangermeme>=0.5`

Downstream users must convert the Illumina SavedModel themselves (license restrictions prevent distributing pre-converted checkpoints).

## Numerical equivalence (`examples/`)

Validated on TERT (*n*=6,006), SFSWAP, and DNAJC9 promoter variants using the `hg38_finetune` and `hg38_mm10_finetune` models. Scores are numerically identical to the original TF/Keras implementation across all comparisons (r=1.0000, MAE=0.0000), including the ensembled score against published PromoterAI output.

Analysis notebooks: `examples/promoter_ism_benchmark/plot_TERT.ipynb`, `examples/promoter_ism_benchmark/plot_SFSWAP.ipynb`, `examples/promoter_ism_benchmark/plot_DNAJC9.ipynb`.

Data files per gene: `{model}.TF.tsv.gz` (original TF scores), `{model}.TORCH.tsv.gz` (PyTorch scores), `{gene}.scores.tsv.gz` (combined, cached).

## PyPI publishing (`.github/workflows/publish.yml`)

Uses PyPI trusted publishing (OIDC) — no API token required. Triggered on GitHub release publication. Environment named `pypi` with `id-token: write` permission.
