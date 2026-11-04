"""The prototype DeepLIFT/SHAP engine must reproduce tangermeme's deep_lift_shap."""

import pytest
import torch
from torch import nn

pytest.importorskip("tangermeme", reason="tangermeme not installed")

from tangermeme.deep_lift_shap import deep_lift_shap as tm_deep_lift_shap  # noqa: E402

from promoterai_torch._deep_lift_shap_engine import deep_lift_shap  # noqa: E402
from promoterai_torch.architecture import PromoterAI  # noqa: E402


class _Wrapper(nn.Module):
    """The quantity examples/deepliftshap_region.py attributes: mean of one head's output."""

    def __init__(self, model, head=0):
        super().__init__()
        self.model, self.head = model, head

    def forward(self, x):
        return self.model(x.transpose(1, 2))[self.head].mean(dim=(1, 2)).unsqueeze(1)


def _model(num_blocks=8, model_dim=8, output_crop=64, seed=0):
    torch.manual_seed(seed)
    model = PromoterAI(
        num_blocks=num_blocks, model_dim=model_dim, output_dims=[5, 3], output_crop=output_crop
    )
    # Non-trivial BatchNorm statistics and larger weights than the init, so every
    # ReLU's rescale rule and both BatchNorm affine terms matter.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.2, 0.2)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.3)
            m.bias.data.normal_(0, 0.1)
    return model.double().eval()


def _X(n, L, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.eye(4)[torch.randint(0, 4, (n, L), generator=g)].transpose(1, 2).contiguous()


@pytest.mark.parametrize(
    "num_blocks, model_dim, L, output_crop, head",
    [
        (8, 8, 256, 64, 0),
        (8, 8, 256, 0, 1),  # no crop, second species head
        (24, 4, 4096, 3072, 0),  # published depth: dilations up to 1024
    ],
)
@pytest.mark.parametrize("ref_chunk", [1, 4])
def test_engine_matches_tangermeme(num_blocks, model_dim, L, output_crop, head, ref_chunk):
    model = _model(num_blocks, model_dim, output_crop)
    X = _X(2, L)
    expected = tm_deep_lift_shap(
        _Wrapper(model, head), X, n_shuffles=6, random_state=0, batch_size=6,
        device="cpu", dtype=torch.float64, warning_threshold=1e9,
    )
    got, deltas = deep_lift_shap(
        model, X, head=head, n_shuffles=6, random_state=0, ref_chunk=ref_chunk,
        return_deltas=True, device="cpu",
    )
    torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-14)
    assert deltas.shape == (2, 6)
    assert float(deltas.max()) < 1e-12  # attributions sum to the output difference


def test_engine_given_references_and_hypothetical():
    model = _model()
    X = _X(1, 256)
    refs = _X(5, 256, seed=1).unsqueeze(0)  # (1, 5, 4, L)
    for hypothetical in (False, True):
        expected = tm_deep_lift_shap(
            _Wrapper(model), X, references=refs, hypothetical=hypothetical,
            device="cpu", dtype=torch.float64, warning_threshold=1e9,
        )
        got = deep_lift_shap(model, X, references=refs, hypothetical=hypothetical, ref_chunk=2)
        torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-14)
