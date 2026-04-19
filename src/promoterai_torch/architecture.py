import torch
import torch.nn as nn


def _dilation_rate(i):
    """Return dilation rate for block i: doubles every 2 blocks (1,1,1,1,2,2,4,4,…)."""
    return max(1, 2 ** (i // 2 - 1))


class MetaFormerBlock(nn.Module):
    def __init__(self, model_dim: int, kernel_size: int, dilation_rate: int):
        """Pre-norm MetaFormer block: depthwise-conv token mixer + FFN channel mixer."""
        super().__init__()
        self.bn1 = nn.BatchNorm1d(model_dim, eps=1e-3)
        self.dw_conv = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size,
            dilation=dilation_rate,
            padding="same",
            groups=model_dim,
        )
        self.bn2 = nn.BatchNorm1d(model_dim, eps=1e-3)
        self.ffn1 = nn.Linear(model_dim, model_dim * 4)
        self.act = nn.ReLU()
        self.ffn2 = nn.Linear(model_dim * 4, model_dim)
        self._init_weights()

    def _init_weights(self):
        """Initialize dw_conv with Glorot-uniform and FFN layers with TruncatedNormal(std=0.01)."""
        nn.init.xavier_uniform_(
            self.dw_conv.weight.view(self.dw_conv.weight.shape[0], -1)
            .unsqueeze(0)
            .squeeze(0)
            .view(self.dw_conv.weight.shape)
        )
        nn.init.trunc_normal_(self.ffn1.weight, std=0.01)
        nn.init.trunc_normal_(self.ffn2.weight, std=0.01)
        nn.init.zeros_(self.ffn1.bias)
        nn.init.zeros_(self.ffn2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply token-mixing and FFN residual branches; x and output are (B, L, model_dim)."""
        x_t = x.transpose(1, 2)  # (B, model_dim, L)
        x_t = self.bn1(x_t)
        x_t = self.dw_conv(x_t)
        x_t = x_t.transpose(1, 2)  # (B, L, model_dim)
        intermediate = x + x_t

        x_t = intermediate.transpose(1, 2)
        x_t = self.bn2(x_t)
        x_t = x_t.transpose(1, 2)
        x_t = self.act(self.ffn1(x_t))
        x_t = self.ffn2(x_t)
        return intermediate + x_t


class OutputHead(nn.Module):
    def __init__(
        self,
        model_dim: int,
        output_dim: int,
        num_blocks: int,
        shortcut_layer_freq: int,
        output_crop: int,
        head_idx: int,
    ):
        """Average relu projections from shortcut layers, then center-crop to output length."""
        super().__init__()
        self.shortcut_indices = list(range(num_blocks, 0, -shortcut_layer_freq))
        self.output_crop = output_crop
        self.projections = nn.ModuleList(
            [nn.Linear(model_dim, output_dim) for _ in self.shortcut_indices]
        )
        self.acts = nn.ModuleList([nn.ReLU() for _ in self.shortcut_indices])
        for proj in self.projections:
            nn.init.trunc_normal_(proj.weight, std=0.01)
            nn.init.zeros_(proj.bias)

    def forward(self, layers: list) -> torch.Tensor:
        """Average shortcut projections from layers list; returns (B, output_length, output_dim)."""
        projected = [
            act(proj(layers[i]))
            for act, proj, i in zip(self.acts, self.projections, self.shortcut_indices)
        ]
        out = torch.stack(projected, dim=0).mean(dim=0)  # (B, L, output_dim)
        if self.output_crop > 0:
            c = self.output_crop // 2
            out = out[:, c:-c, :]
        return out


class PromoterAI(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        model_dim: int,
        output_dims: list,
        kernel_size: int = 5,
        shortcut_layer_freq: int = 4,
        output_crop: int = 0,
    ):
        """Build PromoterAI with given depth, width, per-species output dims, and center crop."""
        super().__init__()
        self.num_blocks = num_blocks
        self.stem = nn.Conv1d(4, model_dim, 1)
        self.stem_act = nn.ReLU()
        nn.init.xavier_uniform_(self.stem.weight)
        nn.init.zeros_(self.stem.bias)

        self.blocks = nn.ModuleList(
            [
                MetaFormerBlock(model_dim, kernel_size, _dilation_rate(i))
                for i in range(num_blocks)
            ]
        )
        self.output_heads = nn.ModuleList(
            [
                OutputHead(
                    model_dim,
                    output_dim,
                    num_blocks,
                    shortcut_layer_freq,
                    output_crop,
                    j,
                )
                for j, output_dim in enumerate(output_dims)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """Run full forward pass; x is (B, L, 4), returns tuple of (B, output_length, output_dim) per head."""
        x_t = x.transpose(1, 2)  # (B, 4, L)
        x_t = self.stem_act(self.stem(x_t))
        layer_0 = x_t.transpose(1, 2)  # (B, L, model_dim)

        layers = [None] * (self.num_blocks + 1)
        layers[0] = layer_0
        for i, block in enumerate(self.blocks):
            layers[i + 1] = block(layers[i])

        return tuple(head(layers) for head in self.output_heads)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return final-block sequence embeddings: (B, L, model_dim)."""
        x_t = x.transpose(1, 2)
        out = x_t.transpose(1, 2)
        out = self.stem_act(self.stem(x_t)).transpose(1, 2)
        for block in self.blocks:
            out = block(out)
        return out


class TwinModel(nn.Module):
    """Wraps PromoterAI for variant effect scoring. Only output_heads[0] is trainable."""

    def __init__(self, base_model: PromoterAI):
        """Freeze all base_model parameters except output_heads[0]."""
        super().__init__()
        self.base_model = base_model
        for param in base_model.parameters():
            param.requires_grad = False
        for param in base_model.output_heads[0].parameters():
            param.requires_grad = True

    def forward(self, x_ref: torch.Tensor, x_alt: torch.Tensor) -> torch.Tensor:
        """Return mean(output_alt - output_ref) averaged over positions and tracks; shape (B,)."""
        out_ref = self.base_model(x_ref)[0]  # (B, L, output_dim)
        out_alt = self.base_model(x_alt)[0]
        diff = (out_alt - out_ref).mean(dim=(1, 2))  # (B,)
        return diff
        # NOTE: np.tanh is applied in score.py, not here
