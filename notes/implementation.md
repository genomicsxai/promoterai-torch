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
isn't worth a custom optimizer to chase. Confirmed in `tests/test_tf_gradient_equivalence.py`
(toy scale) and `tests/test_tf_gradient_equivalence_real.py` (real checkpoint, GPU), which show
the two frameworks' AdamW *mechanics* (bias correction, decoupled weight decay) agree by cosine
similarity once epsilon is isolated out (`epsilon -> 0` on both sides), and that even at the real
`epsilon=1e-7` the step's *direction* still agrees (only its magnitude differs, as expected) —
see `tests/gradient_comparison_utils.py` for the cosine-similarity/relative-L2/pass-rate
comparison methodology, adapted from alphagenome-pytorch's JAX-comparison test suite.

## Multi-species loss must use a soft zero, not a hard skip (`src/promoterai_torch/train.py`)

Illumina's `tfrecords.py` handles multi-species batches (e.g. human + mouse) with a
*soft* zero, not by excluding the inactive species from the loss:

```python
y = tuple(y if sw else [[0.]] for sw in sample_weight)
sample_weight = tuple(sw * tf.reduce_max(x) for sw in sample_weight)
```

The inactive species' target is a dummy placeholder (broadcasting to zero against the
real prediction shape) and its `sample_weight` is `0`, so its loss term is still
computed and stays in the graph — it just evaluates to zero, and via the chain rule
(differentiating a term multiplied by the *constant* `0`) so does its gradient. Because
that gradient is a real, present zero rather than absent, Keras' `AdamW` still runs
`_apply_weight_decay` on that species' output head every batch (a separate,
gradient-independent `variable -= variable * weight_decay * lr` step in the optimizer's
base class), regardless of which species happens to be active that batch.

`compute_loss` originally used a hard Python-level skip instead:

```python
if y_true.shape[-1] == 1:
    continue  # dummy target for non-matching species
```

`continue`-ing meant that species' predictions never entered the loss graph at all, so
PyTorch's autograd left `.grad` as `None` (not zero) for that head's parameters, and
`torch.optim.AdamW.step()` skips any parameter whose gradient is `None` — including its
weight decay. Net effect: our port decayed an inactive-that-batch head's parameters
*less often* than Illumina's real training does (only on that head's own active
batches, not every batch), a real behavioral divergence, not a rounding artifact
(though at `lr=5e-4, weight_decay=5e-6`, a single batch's decay-only change is only
`~2.5e-9` relative to the variable — often below float32's representable precision
against a typical ~0.01-0.1 weight, so it may not be *visible* step to step; it still
compounds differently over the course of training).

Fixed by replacing the `continue` with the same broadcast-to-zero substitution Illumina
uses (`y_true = torch.zeros_like(y_pred)`), keeping the weighted term in the loss sum
unconditionally. See `test_compute_loss_keeps_dummy_species_in_gradient_graph` in
`tests/test_train.py`, and `tests/keras_pytorch_step.py`'s `weights` parameter /
`tests/test_tf_gradient_equivalence_real.py`'s per-head loop, which verify this
cross-framework (including that the inactive head's parameters change identically on
both sides, whatever that change rounds to).

## Numerical equivalence (`examples/`)

Validated on TERT (*n*=6,006), SFSWAP, and DNAJC9 promoter variants using the `hg38_finetune` and `hg38_mm10_finetune` models. Scores are numerically identical to the original TF/Keras implementation across all comparisons (r=1.0000, MAE=0.0000), including the ensembled score against published PromoterAI output.

Analysis notebooks: `examples/promoter_ism_benchmark/plot_TERT.ipynb`, `examples/promoter_ism_benchmark/plot_SFSWAP.ipynb`, `examples/promoter_ism_benchmark/plot_DNAJC9.ipynb`.

Data files per gene: `{model}.TF.tsv.gz` (original TF scores), `{model}.TORCH.tsv.gz` (PyTorch scores), `{gene}.scores.tsv.gz` (combined, cached).

## PyPI publishing (`.github/workflows/publish.yml`)

Uses PyPI trusted publishing (OIDC) — no API token required. Triggered on GitHub release publication. Environment named `pypi` with `id-token: write` permission.
