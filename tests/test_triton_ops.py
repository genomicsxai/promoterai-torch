import pytest
import torch

from promoterai_torch.architecture import MetaFormerBlock
from promoterai_torch.triton_ops import (
    triton_available,
    triton_dw_conv_supported,
)

cuda_triton_available = triton_available() and torch.cuda.is_available()
requires_cuda_triton = pytest.mark.skipif(
    not cuda_triton_available,
    reason="requires triton installed and a CUDA device",
)


def test_triton_dw_conv_unsupported_on_cpu():
    # Triton kernel compilation targets CUDA only; CPU is never "supported",
    # independent of whether triton itself happens to be importable.
    assert triton_dw_conv_supported("cpu", None) is False


def test_invalid_dw_conv_backend_rejected():
    with pytest.raises(ValueError):
        MetaFormerBlock(model_dim=8, kernel_size=3, dilation_rate=1, dw_conv_backend="bogus")


def test_forced_triton_backend_raises_without_support():
    block = MetaFormerBlock(model_dim=8, kernel_size=3, dilation_rate=1, dw_conv_backend="triton")
    x = torch.randn(2, 16, 8)  # CPU tensor: triton is never "supported" here
    with pytest.raises(RuntimeError):
        block(x)


def test_auto_backend_matches_torch_backend_on_cpu():
    # On CPU, "auto" must silently fall back to the plain nn.Conv1d path, so the
    # two backends should be numerically identical (both run the torch path).
    torch.manual_seed(0)
    auto_block = MetaFormerBlock(model_dim=8, kernel_size=5, dilation_rate=2, dw_conv_backend="auto")
    torch_block = MetaFormerBlock(model_dim=8, kernel_size=5, dilation_rate=2, dw_conv_backend="torch")
    torch_block.load_state_dict(auto_block.state_dict())
    auto_block.eval()
    torch_block.eval()

    x = torch.randn(2, 32, 8)
    with torch.no_grad():
        out_auto = auto_block(x)
        out_torch = torch_block(x)
    assert torch.allclose(out_auto, out_torch)


@requires_cuda_triton
@pytest.mark.parametrize("kernel_size", [3, 4, 5, 6])
@pytest.mark.parametrize("dilation", [1, 2, 4])
def test_triton_forward_matches_conv1d(kernel_size, dilation):
    from promoterai_torch.triton_ops import depthwise_dilated_conv1d_triton

    torch.manual_seed(0)
    device = torch.device("cuda")
    N, C, L = 3, 20, 129  # C not a power of 2, on purpose
    conv = torch.nn.Conv1d(
        C, C, kernel_size, dilation=dilation, padding="same", groups=C
    ).to(device)
    x = torch.randn(N, C, L, device=device)

    y_ref = conv(x)
    y_triton = depthwise_dilated_conv1d_triton(x, conv.weight, conv.bias, dilation)
    assert torch.allclose(y_triton, y_ref, atol=1e-4, rtol=1e-4)


@requires_cuda_triton
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("dilation", [1, 4])
def test_triton_backward_matches_conv1d(kernel_size, dilation):
    from promoterai_torch.triton_ops import depthwise_dilated_conv1d_triton

    torch.manual_seed(0)
    device = torch.device("cuda")
    N, C, L = 3, 20, 129
    conv = torch.nn.Conv1d(
        C, C, kernel_size, dilation=dilation, padding="same", groups=C
    ).to(device)
    x_ref = torch.randn(N, C, L, device=device, requires_grad=True)
    x_triton = x_ref.detach().clone().requires_grad_(True)

    y_ref = conv(x_ref)
    dy = torch.randn_like(y_ref)
    y_ref.backward(dy)

    weight_triton = conv.weight.detach().clone().requires_grad_(True)
    bias_triton = conv.bias.detach().clone().requires_grad_(True)
    y_triton = depthwise_dilated_conv1d_triton(x_triton, weight_triton, bias_triton, dilation)
    y_triton.backward(dy)

    assert torch.allclose(x_triton.grad, x_ref.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(weight_triton.grad, conv.weight.grad, atol=1e-3, rtol=1e-3)
    assert torch.allclose(bias_triton.grad, conv.bias.grad, atol=1e-4, rtol=1e-4)


@requires_cuda_triton
def test_metaformer_block_triton_matches_torch_backend():
    torch.manual_seed(0)
    device = torch.device("cuda")
    torch_block = MetaFormerBlock(
        model_dim=16, kernel_size=5, dilation_rate=2, dw_conv_backend="torch"
    ).to(device)
    triton_block = MetaFormerBlock(
        model_dim=16, kernel_size=5, dilation_rate=2, dw_conv_backend="triton"
    ).to(device)
    triton_block.load_state_dict(torch_block.state_dict())
    torch_block.eval()
    triton_block.eval()

    x = torch.randn(2, 64, 16, device=device)
    with torch.no_grad():
        out_torch = torch_block(x)
        out_triton = triton_block(x)
    assert torch.allclose(out_triton, out_torch, atol=1e-4, rtol=1e-4)
