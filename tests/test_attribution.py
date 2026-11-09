"""promoterai_torch.attribution.deep_lift_shap must reproduce tangermeme's deep_lift_shap."""

import pytest
import torch
from torch import nn

pytest.importorskip("tangermeme", reason="tangermeme not installed")

from tangermeme.deep_lift_shap import deep_lift_shap as tm_deep_lift_shap  # noqa: E402

from promoterai_torch import attribution as engine  # noqa: E402
from promoterai_torch.attribution import deep_lift_shap  # noqa: E402
from promoterai_torch.architecture import PromoterAI  # noqa: E402


class _Wrapper(nn.Module):
    """The quantity examples/deepliftshap_region.py attributes: mean of one head's output."""

    def __init__(self, model, head=0, tracks=None):
        super().__init__()
        self.model, self.head, self.tracks = model, head, tracks

    def forward(self, x):
        out = self.model(x.transpose(1, 2))[self.head]
        if self.tracks is not None:
            out = out[..., self.tracks]
        return out.mean(dim=(1, 2)).unsqueeze(1)


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
        return_deltas=True, device="cpu", batch_size=1,
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


def _tm(model, X, **kw):
    head, tracks = kw.pop("head", 0), kw.pop("tracks", None)
    return tm_deep_lift_shap(
        _Wrapper(model, head, tracks), X, device="cpu", dtype=torch.float64,
        warning_threshold=1e9, **kw,
    )


def test_engine_tracks_subset():
    model = _model()
    X = _X(2, 256)
    for tracks in ([2], [0, 3, 4]):
        expected = _tm(model, X, tracks=tracks, n_shuffles=4, random_state=3)
        got = deep_lift_shap(model, X, tracks=tracks, n_shuffles=4, random_state=3)
        torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-14)
    torch.testing.assert_close(
        deep_lift_shap(model, X, tracks=[-1, 0], n_shuffles=2, random_state=0),
        deep_lift_shap(model, X, tracks=[4, 0], n_shuffles=2, random_state=0),
    )
    for bad in ([5], [-6], [4, -1], []):  # out of range, aliased duplicate, empty
        with pytest.raises(ValueError, match="tracks"):
            deep_lift_shap(model, X, tracks=bad, n_shuffles=2)


@pytest.mark.parametrize("batch_size, ref_chunk", [(2, None), (2, 4), (3, 6), (1, 5), (None, None)])
def test_engine_batching_matches_one_at_a_time(batch_size, ref_chunk):
    """Several sequences per step, and reference chunks that don't divide n_shuffles."""
    model = _model()
    X = _X(5, 256, seed=2)
    expected, d_expected = deep_lift_shap(
        model, X, n_shuffles=6, random_state=0, batch_size=1, ref_chunk=1, return_deltas=True
    )
    got, d = deep_lift_shap(
        model, X, n_shuffles=6, random_state=0, batch_size=batch_size, ref_chunk=ref_chunk,
        return_deltas=True,
    )
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-16)
    assert d.shape == d_expected.shape == (5, 6) and float(d.max()) < 1e-12


def test_plan(monkeypatch):
    cpu, f32, GiB = torch.device("cpu"), torch.float32, 1 << 30
    full = _model(num_blocks=24, model_dim=1024, output_crop=0).float()
    L, M = 20480, 20480 * 1024 * 4
    # Published scale: 4 references a step saturate (an A100 80GB gets 4)...
    assert engine._plan(full, L, 10, 20, f32, cpu, 80 * GiB) == (1, 4)
    # ... a 12 GB card only fits 2 in its budget.
    assert engine._plan(full, L, 10, 20, f32, cpu, 10 * GiB) == (1, 2)
    with pytest.warns(UserWarning, match="budget"):
        assert engine._plan(full, L, 1, 20, f32, cpu, M) == (1, 1)
    # Narrow model: references per step up to the memory budget...
    narrow = _model(num_blocks=24, model_dim=64, output_crop=0).float()
    assert engine._plan(narrow, L, 100, 20, f32, cpu, 2 * GiB) == (1, 9)
    assert engine._plan(narrow, L, 100, 20, f32, cpu, GiB // 2)[1] == 1
    # ... and once all of them fit, sequences are batched to saturation or the budget.
    assert engine._plan(narrow, L, 100, 20, f32, cpu, 10 * GiB) == (2, 20)
    assert engine._plan(narrow, 2048, 100, 20, f32, cpu, 10 * GiB) == (24, 20)
    assert engine._plan(narrow, 512, 3, 20, f32, cpu, 10 * GiB) == (3, 20)
    # torch.compile (here: CPU) is bounded by the eager model; only the TorchScript
    # fallback on capability < 7.0 GPUs budgets more per step.
    eager = engine._plan(narrow, L, 100, 20, f32, cpu, GiB)
    assert engine._plan(narrow, L, 100, 20, f32, cpu, GiB, compile=True) == eager
    monkeypatch.setattr(engine, "_torchscript_fallback", lambda device: True)
    assert engine._plan(narrow, L, 100, 20, f32, cpu, GiB, compile=True)[1] == 3 < eager[1]
    assert engine._plan(narrow, L, 100, 20, f32, cpu, GiB) == eager


@pytest.mark.parametrize(
    "K, k, expected",
    [
        (20, 4, [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20)]),
        (21, 4, [(0, 4), (4, 8), (8, 12), (12, 16), (16, 19), (19, 21)]),  # no lone reference
        (5, 2, [(0, 2), (2, 5)]),
        (6, 4, [(0, 4), (4, 6)]),
        (3, 1, [(0, 1), (1, 2), (2, 3)]),
        (1, 4, [(0, 1)]),
    ],
)
def test_chunks(K, k, expected):
    assert engine._chunks(K, k) == expected


def test_engine_retries_smaller_after_oom(monkeypatch):
    model = _model()
    X = _X(3, 256)
    expected = deep_lift_shap(model, X, n_shuffles=4, random_state=0, batch_size=1, ref_chunk=1)
    real, calls = engine._step, []

    def flaky(*args):
        B, k = args[7].shape[:2]
        calls.append((B, k))
        if B * k > 2:
            raise torch.OutOfMemoryError("fake")
        return real(*args)

    monkeypatch.setattr(engine, "_step", flaky)
    with pytest.warns(UserWarning, match="retrying"):
        got = deep_lift_shap(model, X, n_shuffles=4, random_state=0, batch_size=3, ref_chunk=4)
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-16)
    assert calls[0] == (3, 4) and (1, 2) in calls


def test_engine_compiled_matches_eager():
    pytest.importorskip("torch._inductor")
    model = _model()
    X = _X(2, 256)
    eager = deep_lift_shap(model, X, n_shuffles=3, random_state=0, ref_chunk=2)
    compiled = deep_lift_shap(model, X, n_shuffles=3, random_state=0, ref_chunk=2, compile=True)
    torch.testing.assert_close(compiled, eager, rtol=1e-9, atol=1e-13)


def test_rescale_variants_agree():
    """In-place (eager), functional (torch.compile) and TorchScript (pre-Volta GPUs) rules."""
    import warnings

    g = torch.Generator().manual_seed(0)
    a_x = torch.randn(2, 1, 64, 8, generator=g, dtype=torch.float64)
    a_r = torch.randn(2, 3, 64, 8, generator=g, dtype=torch.float64)
    a_r[:, 0, :8] = a_x[:, 0, :8] + 1e-8  # the plain-gradient fallback
    G = torch.randn(2, 3, 64, 8, generator=g, dtype=torch.float64)
    expected = engine._rescale(G, a_x, a_r)
    torch.testing.assert_close(engine._rescale_(G.clone(), a_x, a_r.clone()), expected)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scripted = torch.jit.script(engine._rescale)
    torch.testing.assert_close(scripted(G, a_x, a_r), expected)


def test_input_validation():
    model = _model()
    with pytest.raises(ValueError, match="one-hot"):
        deep_lift_shap(model, _X(1, 256).transpose(1, 2))
    with pytest.raises(ValueError, match="head"):
        deep_lift_shap(model, _X(1, 256), head=2)
    with pytest.raises(ValueError, match="references"):
        deep_lift_shap(model, _X(2, 256), references=_X(3, 256).unsqueeze(0))
    with pytest.raises(TypeError, match="PromoterAI"):
        deep_lift_shap(_Wrapper(model), _X(1, 256))
