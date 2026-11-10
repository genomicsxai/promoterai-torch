# Choosing a PyTorch build

`pip install promoterai-torch` and `uv add promoterai-torch` take `torch` from
PyPI. On Linux that build uses CUDA 13.0 and supports NVIDIA GPUs of compute
capability 7.5 (Turing) and newer. For an older GPU, or a CPU-only machine,
install a different build from PyTorch's own package index:

| GPU | Compute capability | Build |
| --- | --- | --- |
| Maxwell, Pascal, Volta (e.g. GTX 10xx, TITAN X/Xp, P100, V100) | 5.0–7.0 | CUDA 12.6 (`cu126`) |
| Turing and newer (RTX 20xx+, T4, A100, H100, Blackwell) | 7.5+ | PyPI default, or `cu130` |
| None | — | `cpu` |

Check your GPU's compute capability with
`nvidia-smi --query-gpu=name,compute_cap --format=csv`. A build that doesn't
support your GPU fails at the first CUDA operation with
"no kernel image is available for execution on the device".

With pip, install torch from the matching index first, then the package:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or cu130, cpu
pip install promoterai-torch
```

In a clone of this repository, uv selects the build with a dependency group
(`cu126`, `cu130` or `cpu`; use at most one):

```sh
uv sync --group cu126                          # CUDA 12.6 build, for Pascal etc.
uv sync --group dev --extra train --group cu126
uv run --group cu126 pytest tests/             # pass the same group to uv run
```

`uv sync` and `uv run` without the group switch the environment back to the
PyPI build (and download it), so pass the group every time. To keep several
builds side by side, give each its own environment:

```sh
export UV_PROJECT_ENVIRONMENT=/path/to/venv-cu126
uv sync --group dev --group cu126
uv run --group cu126 promoterai-torch score ...
```

The CUDA 12.6 line is PyTorch's legacy build: its last release is older than
the newest PyPI torch, so a `cu126` environment may resolve an older torch
version than the default one.

For development installs (test dependencies, the `tf` group for the
cross-framework tests), see [CONTRIBUTING.md](../CONTRIBUTING.md).
