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
