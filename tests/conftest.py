"""Shared pytest CLI options/fixtures for real-checkpoint, GPU-backed tests.

See tests/test_tf_gradient_equivalence_real.py.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--keras-savedmodel-path",
        default=None,
        help=(
            "Path to a real Illumina PromoterAI Keras SavedModel directory. Required to "
            "run tests/test_tf_gradient_equivalence_real.py; skipped otherwise, since the "
            "model is licensed and isn't distributed with this repo."
        ),
    )
    parser.addoption(
        "--device",
        default=None,
        help="torch device for real-checkpoint tests (default: cuda if available, else cpu)",
    )
    parser.addoption(
        "--gradient-batch-size",
        type=int,
        default=2,
        help="Batch size for tests/test_tf_gradient_equivalence_real.py (default: %(default)s)",
    )
    parser.addoption(
        "--gradient-input-length",
        type=int,
        default=20480,
        help=(
            "Input sequence length for tests/test_tf_gradient_equivalence_real.py "
            "(default: the published model's %(default)s)"
        ),
    )
    parser.addoption(
        "--gradient-output-length",
        type=int,
        default=4096,
        help=(
            "Output track length for tests/test_tf_gradient_equivalence_real.py "
            "(default: the published model's %(default)s)"
        ),
    )


@pytest.fixture(scope="session")
def keras_savedmodel_path(request):
    """Path to a real Keras SavedModel, or skip the test if not provided."""
    path = request.config.getoption("--keras-savedmodel-path")
    if not path:
        pytest.skip(
            "requires --keras-savedmodel-path (a licensed Illumina PromoterAI Keras "
            "SavedModel; not distributed with this repo)"
        )
    return path


@pytest.fixture(scope="session")
def gradient_device(request):
    """torch device for real-checkpoint tests; defaults to cuda when available."""
    import torch

    device = request.config.getoption("--device")
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def gradient_batch_size(request):
    return request.config.getoption("--gradient-batch-size")


@pytest.fixture(scope="session")
def gradient_input_length(request):
    return request.config.getoption("--gradient-input-length")


@pytest.fixture(scope="session")
def gradient_output_length(request):
    return request.config.getoption("--gradient-output-length")
