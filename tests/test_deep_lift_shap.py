"""
Verify DeepLIFT/SHAP convergence deltas are within tolerance after the
architecture was updated to use named nn.ReLU() modules.

Skipped automatically when tangermeme is not installed.
"""

import warnings

import pytest
import torch
from torch import nn

pytest.importorskip("tangermeme", reason="tangermeme not installed")


def _build_model(num_blocks=4, model_dim=8, output_dims=(4,), input_length=20480):
    from promoterai_torch.architecture import PromoterAI

    return PromoterAI(
        num_blocks=num_blocks,
        model_dim=model_dim,
        output_dims=list(output_dims),
        shortcut_layer_freq=4,
        output_crop=0,
    ).eval()


class _Wrapper(nn.Module):
    """Adapt PromoterAI for tangermeme: channels-first in → (B, n_targets) out."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):  # x: (B, 4, L) channels-first
        out = self.model(x.transpose(1, 2))  # PromoterAI wants (B, L, 4)
        return out[0].mean(dim=1)  # (B, output_dim)


def test_deep_lift_shap_random_sequences():
    """Convergence deltas stay below threshold on random one-hot sequences."""
    from tangermeme.deep_lift_shap import deep_lift_shap
    from tangermeme.utils import random_one_hot

    model = _build_model()
    wrapper = _Wrapper(model)

    x = random_one_hot((4, 4, 64), random_state=0).float()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message="Convergence deltas too high", category=RuntimeWarning
        )
        deep_lift_shap(wrapper, x, n_shuffles=3, device="cpu", random_state=0)


def test_deep_lift_shap_convergence_deltas():
    """DeepLIFT/SHAP convergence deltas stay below threshold for the full model."""
    from tangermeme.deep_lift_shap import deep_lift_shap

    torch.manual_seed(0)
    model = _build_model()
    wrapper = _Wrapper(model)

    x = torch.zeros(4, 4, 64)
    x[:, 0, :] = 1.0  # all-A sequences

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message="Convergence deltas too high", category=RuntimeWarning
        )
        deep_lift_shap(wrapper, x, n_shuffles=3, device="cpu", random_state=0)
