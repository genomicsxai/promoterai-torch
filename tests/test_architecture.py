import pytest  # noqa
import torch
from promoterai_torch.architecture import (
    MetaFormerBlock,
    OutputHead,
    PromoterAI,
    TwinModel,
    _dilation_rate,
)


def test_dilation_schedule():
    # max(1, 2 ** (i // 2 - 1)): dilation doubles every 2 blocks after block 3
    expected = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
        6: 4,
        7: 4,
        8: 8,
        9: 8,
        10: 16,
        11: 16,
        12: 32,
        13: 32,
        14: 64,
        15: 64,
        16: 128,
        17: 128,
        18: 256,
        19: 256,
        20: 512,
        21: 512,
        22: 1024,
        23: 1024,
    }
    for i, d in expected.items():
        assert _dilation_rate(i) == d, (
            f"block {i}: expected dilation {d}, got {_dilation_rate(i)}"
        )


def test_metaformer_block_shape():
    B, L, C = 2, 128, 64
    block = MetaFormerBlock(C, kernel_size=5, dilation_rate=1)
    x = torch.randn(B, L, C)
    out = block(x)
    assert out.shape == (B, L, C)


def test_metaformer_residual():
    block = MetaFormerBlock(32, 5, 1)
    block.eval()
    # Zero out all weights so output = input (residual-only path)
    with torch.no_grad():
        block.dw_conv.weight.zero_()
        block.dw_conv.bias.zero_()
        block.ffn1.weight.zero_()
        block.ffn1.bias.zero_()
        block.ffn2.weight.zero_()
        block.ffn2.bias.zero_()
    x = torch.randn(1, 64, 32)
    out = block(x)
    assert torch.allclose(out, x, atol=1e-5)


def test_output_head_shape():
    B, L, C, D = 2, 128, 64, 10
    num_blocks, shortcut_freq, output_crop = 8, 4, 0
    layers = [torch.randn(B, L, C) for _ in range(num_blocks + 1)]
    head = OutputHead(C, D, num_blocks, shortcut_freq, output_crop, head_idx=0)
    out = head(layers)
    assert out.shape == (B, L, D)


def test_output_head_crop():
    B, L, C, D = 2, 128, 64, 10
    num_blocks, shortcut_freq, output_crop = 8, 4, 64
    layers = [torch.randn(B, L, C) for _ in range(num_blocks + 1)]
    head = OutputHead(C, D, num_blocks, shortcut_freq, output_crop, head_idx=0)
    out = head(layers)
    assert out.shape == (B, L - output_crop, D)


def test_promoter_ai_forward():
    B, L = 2, 512
    model = PromoterAI(num_blocks=4, model_dim=32, output_dims=[10, 8], output_crop=0)
    x = torch.zeros(B, L, 4)
    x[:, :, 0] = 1.0  # all-A sequence
    outputs = model(x)
    assert len(outputs) == 2
    assert outputs[0].shape == (B, L, 10)
    assert outputs[1].shape == (B, L, 8)


def test_promoter_ai_with_crop():
    B, input_len, output_len = 2, 512, 256
    model = PromoterAI(
        num_blocks=4, model_dim=32, output_dims=[10], output_crop=input_len - output_len
    )
    x = torch.zeros(B, input_len, 4)
    x[:, :, 1] = 1.0
    outputs = model(x)
    assert outputs[0].shape == (B, output_len, 10)


def test_shortcut_indices():
    num_blocks, freq = 24, 4
    expected = list(range(24, 0, -4))  # [24, 20, 16, 12, 8, 4]
    model = PromoterAI(
        num_blocks=num_blocks, model_dim=32, output_dims=[10], shortcut_layer_freq=freq
    )
    actual = model.output_heads[0].shortcut_indices
    assert actual == expected, f"got {actual}, expected {expected}"


def test_twin_model_trainable_params():
    model = PromoterAI(num_blocks=4, model_dim=32, output_dims=[10, 8])
    twin = TwinModel(model)
    trainable = [n for n, p in twin.named_parameters() if p.requires_grad]
    # Only output_heads[0] params should be trainable
    assert all("output_heads.0" in n for n in trainable), (
        f"Non-output0 params trainable: {trainable}"
    )
    assert len(trainable) > 0


def test_twin_model_forward():
    B, L = 2, 256
    model = PromoterAI(num_blocks=4, model_dim=32, output_dims=[10], output_crop=0)
    twin = TwinModel(model)
    x_ref = torch.zeros(B, L, 4)
    x_ref[:, :, 0] = 1.0
    x_alt = torch.zeros(B, L, 4)
    x_alt[:, :, 1] = 1.0
    diff = twin(x_ref, x_alt)
    assert diff.shape == (B,)


def test_twin_model_same_input_zero_diff():
    B, L = 1, 128
    model = PromoterAI(num_blocks=4, model_dim=32, output_dims=[10], output_crop=0)
    model.eval()
    twin = TwinModel(model)
    twin.eval()
    x = torch.zeros(B, L, 4)
    x[:, :, 0] = 1.0
    diff = twin(x, x)
    assert torch.allclose(diff, torch.zeros(B), atol=1e-6)


def test_twin_model_train_keeps_backbone_batchnorm_frozen():
    model = PromoterAI(num_blocks=4, model_dim=8, output_dims=[3], output_crop=0)
    twin = TwinModel(model)

    twin.train()

    assert twin.training is True
    assert model.training is False
    assert model.blocks[0].bn1.training is False
    assert model.output_heads[0].training is True
