# Contributing

## Development setup

Clone the repository and install local development dependencies with `uv`
dependency groups:

```sh
uv sync --group dev
uv run pytest tests/ -v
```

Use the narrower groups when you only need one workflow:

```sh
uv sync --group test
uv run pytest tests/ -v

uv sync --group publish
uv run python -m build
uv run python -m twine check dist/*
```

For a `pip`-only editable install, install the package and development tools
directly:

```sh
python -m pip install -e .
python -m pip install pytest h5py pybigtools tangermeme build twine
python -m pytest tests/ -v
```

## Pull requests

Run the test suite before opening a PR, and keep changes scoped to a single
concern where possible.

If your change touches the model architecture, training loop, optimizer
setup, or the Keras/PyTorch weight converter, also run the cross-framework
equivalence tests locally before merging:

```sh
uv sync --group dev --extra convert
uv run pytest tests/test_convert.py tests/test_tf_gradient_equivalence.py -v
```

These require `tensorflow`/`tf-keras` (the `convert` extra) and are skipped
by default in per-PR CI, so they won't catch a regression there. The
`tf-equivalence.yml` workflow runs them weekly as a final failsafe against TF
or PyTorch API drift — not a substitute for running them yourself when your
PR touches equivalence-sensitive code.

If you have a real Illumina PromoterAI Keras SavedModel and a GPU, also run
the same comparison at full published scale (`num_blocks=24`, `model_dim=1024`)
rather than `test_tf_gradient_equivalence.py`'s toy 8-block model:

```sh
uv run pytest tests/test_tf_gradient_equivalence_real.py -v -s \
    --keras-savedmodel-path /path/to/promoterai_keras_model \
    --device cuda --gradient-batch-size 2
```

This is skipped by default (including in CI, which has no GPU runner or
licensed SavedModel) — see `tests/conftest.py` for the available options.
