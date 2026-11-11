"""Shared pytest CLI options/fixtures for real-checkpoint, GPU-backed tests.

See tests/test_tf_gradient_equivalence_real.py.
"""

import importlib.util

import pytest


def pytest_configure(config):
    """Hide GPUs from TensorFlow before any test touches it.

    Every Keras side in these tests runs on the CPU on purpose (see
    tests/keras_pytorch_step.py:force_tf_cpu), but TensorFlow only accepts a
    visible-device change before it initializes, and the first TF op anywhere in
    the session (building a toy Keras model, or an earlier test such as
    test_convert) initializes it on the GPU. A later force_tf_cpu() then fails with
    "Visible devices cannot be modified after being initialized". Hiding the GPUs
    here, once, keeps that later call a no-op. PyTorch is unaffected. Skipped when
    TensorFlow isn't installed (the `tf` group), so it costs nothing otherwise.
    """
    if importlib.util.find_spec("tensorflow") is None:
        return
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")


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
