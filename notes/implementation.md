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

The "epsilon isolated out" (`param_delta_tiny_eps`, `tiny_eps=1e-10`) check is only *asserted*
in the toy-scale test, not the real-checkpoint one, where it's diagnostic-only (printed, not
gated). At real scale, against a genuine from-scratch base checkpoint, it failed for a
scale-dependent reason unrelated to any mechanics disagreement: `train.py`'s
`clip_grad_norm_per_parameter` clips each *whole tensor's* gradient norm to `max_norm=1e-4`, so
for a huge tensor (an FFN weight is `model_dim x 4*model_dim`, ~4M+ elements at real scale vs. 64
in the toy model) that norm budget spreads thin — average per-element gradient lands around
`1e-4/sqrt(4e6) ~ 5e-8`, giving `sqrt(v) ~ 1.5e-9` for a typical element, only ~15x larger than
`tiny_eps=1e-10`. For the smaller-than-typical elements inevitable in a heavy-tailed gradient
distribution, `epsilon` can end up *larger* than `sqrt(v)` for that element — exactly where
Keras' and PyTorch's differing epsilon-placement formulas diverge most, since that's the same
mechanism as the real-`epsilon=1e-7` case above, just triggered by the isolation trick's own
(otherwise arbitrary) `tiny_eps` choice instead of the real epsilon. Meanwhile raw/post-clip
gradients, the real-epsilon `param_delta` direction check, and BatchNorm running stats all passed
cleanly at real scale — those are what actually validates `train.py`'s production mechanics; this
one check's failure is a confound in the diagnostic itself, not evidence against them.

The same confound hit `tests/test_tf_gradient_equivalence_finetune_real.py` too, on a
real fine-tuned checkpoint's output-head projections (only 12 tensors, moderate
size, but `finetune.py`'s `clip_norm=1.0` is much looser than `train.py`'s `1e-4` --
the dilution mechanism above isn't why here; it's simpler: some of the six shortcut
projections just have smaller raw gradient magnitude than others, and for those,
`tiny_eps=1e-10` still isn't negligible relative to their `sqrt(v)`). Fixed the same
way: `param_delta_tiny_eps` is diagnostic-only (printed) in both real-checkpoint
tests, and still asserted in both toy-scale tests, where the confound doesn't apply.

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

## Multi-species batches must be species-homogeneous (`src/promoterai_torch/dataset.py`)

Illumina's `train.py` combines per-species `tf.data` streams with
`tf.data.Dataset.sample_from_datasets([ds.repeat() for ds in datasets], weights=...)`,
where each per-species stream is already `.batch(batch_size)`'d
(`tfrecords.py`'s `make_dataset`) *before* being combined. That means
`sample_from_datasets` picks one whole batch from one species at a time --
every training batch is single-species, never a mix.

`build_weighted_dataloader` originally combined all species into one
`ConcatDataset` and drew individual **rows** from a single
`WeightedRandomSampler` over the whole thing, so a batch could (and, with
`replacement=True` sampling, almost always did) mix rows from different
species. Since each species dataset returns a real-shaped target for its own
head and a `(1, 1)` dummy placeholder for every other head (see the soft-zero
section below), mixing species within a batch means the same tuple position
holds different shapes across rows in that batch -- `torch.stack` in the
default `collate_fn` then crashes outright:

```
RuntimeError: stack expects each tensor to be equal size, but got [50, 5] at entry 0 and [1, 1] at entry 5
```

So multi-species training (any run with `--hdf5_nonhuman_folders` set)
crashed on essentially its first batch -- this made the soft-zero
`compute_loss` fix below unreachable in practice, since no mixed-species
batch ever survived collation to reach it.

Fixed with `_SpeciesBatchSampler`, a batch-level sampler that picks one
species per batch (weighted by that species' dataset size, matching the old
per-row weighting exactly since `batch_size` is the same constant across
species) and draws the whole batch's indices from only that species. See
`test_multi_species_batches_are_species_homogeneous` in `tests/test_dataset.py`.

## Boundary TSS rows should stay in the dataset as zero rows, not be dropped (`src/promoterai_torch/preprocess.py`)

Illumina's `generator.py` pre-allocates a row per TSS and, when a TSS is too
close to a chromosome edge for a full window (`len(seq) < input_length`),
`continue`s without filling it in -- the row stays in the dataset as an
all-zero row, which `_prepare_sample`'s `x.max() == 0` check later
zero-weights (same mechanism as the multi-species dummy-target case above).

`preprocess_chrom` instead used a hard `continue` that skipped appending the
row entirely, silently shrinking that chromosome's sample count. Low
severity (only affects a handful of near-edge TSS, and only shifts
`steps_per_epoch`/inter-species sampling ratios slightly) but still a real
fidelity gap, and it made the `x_crop.max() == 0` masking path in
`_prepare_sample` dead code in practice for real HDF5-backed data. Fixed by
appending an explicit all-zero `(input_length, 4)` / `(output_length,
n_tracks)` row instead of skipping it -- see
`test_preprocess_keeps_boundary_tss_as_zero_row_matching_tf_generator` in
`tests/test_preprocess.py`.

## Cross-framework test harness must use convert_tf_weights' real species_order, not a name heuristic (`tests/keras_pytorch_step.py`)

`tests/test_tf_gradient_equivalence_real.py`'s first run against a real
human+mouse checkpoint produced a forward-pass loss mismatch far outside
floating-point noise (`loss_pt=43.76` vs `loss_keras=101.31`, head 0 active).
Root cause: `normalize_keras_outputs` ordered a multi-species Keras model's
output dict by a "human/hg38-first, mouse/mm10-second" name heuristic on the
dict's own keys, but `convert_tf_weights`' `species_order` (which actually
determines `output_heads`' index order in the converted PyTorch model) is
just the *first-appearance order of `shortcut_{species}{N}` weights in the
SavedModel's own weight list* -- not necessarily human-first. AGENTS.md's
species-ordering convention is a guideline for authoring new checkpoints,
not something `convert_tf_weights` enforces or could enforce (it only reads
whatever order the SavedModel's weights happen to be in).

When the heuristic's guessed order didn't match `species_order`, the test
paired PyTorch's head 0 (species A) prediction/target against Keras' head 0
under the heuristic's ordering (species B) -- a comparison between two
unrelated heads, not a numerical-precision issue, hence the large mismatch
with no shape error to catch it (both heads' predictions/targets are
shape-compatible, just semantically different species).

Fixed by having `convert_tf_weights` save `species_order` into the
checkpoint's `args` dict, and `normalize_keras_outputs`/`run_single_step`
prefer that ground truth (via a new `species_order` parameter) over the name
heuristic, falling back to the heuristic only when no `species_order` is
given or it doesn't match the dict's actual keys (e.g. a checkpoint
converted before this fix). See `tests/test_keras_pytorch_step.py` and
`test_convert_multi_species`'s `species_order` assertion in
`tests/test_convert.py`.

## Fine-tuned checkpoints need a different gradient-equivalence test than from-scratch ones

`tests/test_tf_gradient_equivalence_real.py` assumes `train.py`'s scenario: every
parameter trainable, every BatchNorm layer computing live batch statistics in
train mode, symmetrically on both frameworks. Running it against an actually
fine-tuned checkpoint (e.g. `hg38_finetune`, `hg38_mm10_finetune`) produced a
large, deterministic forward-pass mismatch that survived every precision
intervention tried (species_order fix, disabling TF32, disabling oneDNN via
`TF_ENABLE_ONEDNN_OPTS=0` -- the last of these *did* measurably shrink one
head's discrepancy, confirming oneDNN's operation-reordering was a real but
partial contributor, while the other head's loss stayed bit-for-bit identical
across every attempt, ruling out precision as its cause).

The actual root cause: a fine-tuned SavedModel has most of its variables
marked non-trainable (the backbone, and every non-primary species' output
head) -- confirmed by `keras_model.trainable_variables` containing exactly 12
tensors, i.e. exactly one output head's worth (6 shortcut projections x
weight+bias). Keras forces a `BatchNormalization` layer into inference mode
(fixed running stats) whenever its `trainable` attribute is `False`,
*regardless of the `training=True` argument* passed to the model call. A
blanket `pt_model.train()` on the PyTorch side has no such exception -- every
BatchNorm layer computes live batch statistics no matter what. So the two
sides were normalizing the backbone completely differently: fixed stats
(Keras) vs. freshly-computed batch stats on synthetic random input (PyTorch)
-- a genuine structural mismatch, not a rounding-order artifact, hence its
complete immunity to TF32/oneDNN toggling.

Fixed by adding a *separate* test, `tests/test_tf_gradient_equivalence_finetune_real.py`
(toy-scale counterpart: `tests/test_tf_gradient_equivalence_finetune.py`), using
a new `run_single_finetune_step` in `tests/keras_pytorch_step.py` that mirrors
`finetune.py`/`TwinModel` exactly instead: one ref/alt-diff AdamW(clipnorm=1.0)
step through `output_heads[0]` only, with `TwinModel.train()`'s existing
`base_model.eval()` / `output_heads[0].train()` split providing the matching
freeze on the PyTorch side (no special-casing needed on the Keras side --
its own non-trainable-layer behavior handles it automatically). This test also
asserts the frozen backbone's BatchNorm running stats are bit-identical
before/after on both sides, as a direct check that the freeze is actually
taking effect.

`test_tf_gradient_equivalence_real.py` remains the right test for a genuine
from-scratch/fully-trainable checkpoint (test against one of those instead of
a fine-tuned checkpoint if you have it) -- the two tests are not
interchangeable, and pointing either at the wrong kind of checkpoint will
produce a spurious mismatch that looks like a porting bug but isn't.

## cosine_threshold=1.0 is the wrong tool for asserting "nothing changed"

`tests/test_tf_gradient_equivalence_finetune_real.py`'s frozen-backbone check
(`bn_unchanged_keras`/`bn_unchanged_pt`) originally used `assert_pass_rate`
with `cosine_threshold=1.0, rel_l2_tol=1e-9` -- and failed on a real
fine-tuned checkpoint, with ~26% of 96 backbone BatchNorm tensors "failing"
despite the printed top offenders all showing `cosine=1.0000, rel_l2=0.0000%,
max_diff=0` (i.e. displaying identically to the tensors that passed).

Root cause: cosine similarity is computed via `dot(a, b) / (norm(a) *
norm(b))`, which involves a `sqrt` and a division -- `sqrt(x)**2` doesn't
always exactly recover `x` in floating point, so even two **bit-identical**
arrays can compute to a cosine of `0.999999999999998` or `1.0000000000000002`
depending on which way that ~1e-16-level rounding happens to fall. A
`cosine_threshold=1.0` pass-rate check is therefore inherently, spuriously
flaky in either direction, regardless of whether anything actually changed --
`assert_pass_rate`/cosine similarity is designed for "are two *different*
frameworks' outputs reasonably close," not "are these two snapshots exactly
identical."

Fixed with a new `assert_exact_match` in `tests/gradient_comparison_utils.py`,
which checks `max_abs_diff == 0` directly (a field `ComparisonResult` already
computes) instead of going through cosine/rel_l2 at all. Use this whenever
asserting a buffer was untouched (e.g. a frozen BatchNorm running stat); keep
`assert_pass_rate` for comparing two frameworks' genuinely-independent
computations.

## Numerical equivalence (`examples/`)

Validated on TERT (*n*=6,006), SFSWAP, and DNAJC9 promoter variants using the `hg38_finetune` and `hg38_mm10_finetune` models. Scores are numerically identical to the original TF/Keras implementation across all comparisons (r=1.0000, MAE=0.0000), including the ensembled score against published PromoterAI output.

Analysis notebooks: `examples/promoter_ism_benchmark/plot_TERT.ipynb`, `examples/promoter_ism_benchmark/plot_SFSWAP.ipynb`, `examples/promoter_ism_benchmark/plot_DNAJC9.ipynb`.

Data files per gene: `{model}.TF.tsv.gz` (original TF scores), `{model}.TORCH.tsv.gz` (PyTorch scores), `{gene}.scores.tsv.gz` (combined, cached).

## PyPI publishing (`.github/workflows/publish.yml`)

Uses PyPI trusted publishing (OIDC) — no API token required. Triggered on GitHub release publication. Environment named `pypi` with `id-token: write` permission.
