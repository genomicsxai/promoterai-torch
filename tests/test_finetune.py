import subprocess
import sys

import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from promoterai_torch.architecture import PromoterAI, TwinModel
from promoterai_torch.finetune import (
    DistributedSliceSampler,
    _run_epoch,
    build_finetune_optimizer,
    filter_finetune_variants,
    load_finetune_checkpoint,
    resolve_finetune_epoch_sizes,
    resolve_finetune_resume_checkpoint,
    save_finetune_checkpoint,
)
from promoterai_torch.utils import autocast_context, load_pretrained


def test_finetune_run_epoch_uses_standard_backward_path():
    model = _TinyTwinModel()
    loader = DataLoader(_TinyVariantDataset(), batch_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    initial_scale = model.scale.detach().clone()

    loss = _run_epoch(model, loader, optimizer, torch.device("cpu"))

    assert loss > 0.0
    assert not torch.equal(model.scale.detach(), initial_scale)


def test_filter_finetune_variants_reports_retained_counts():
    df = pd.DataFrame(
        [
            _variant_row("GENE1", in_cds=0, spliceai=0.01, p_under=0.001),
            _variant_row("GENE1", in_cds=0, spliceai=0.01),
            _variant_row("GENE2", in_cds=1, spliceai=0.01, p_over=0.001),
            _variant_row("GENE3", in_cds=0, spliceai=0.10, p_under=0.001),
            _variant_row("GENE4", in_cds=0, spliceai=0.01),
        ]
    )

    retained, stats = filter_finetune_variants(df)

    assert retained["gene"].tolist() == ["GENE1", "GENE1"]
    assert stats == {
        "total_variants": 5,
        "eligible_variants": 3,
        "outlier_variants": 1,
        "outlier_genes": 1,
        "retained_variants": 2,
    }


def test_finetune_fp16_scaler_unscales_before_clipping(monkeypatch):
    events = []
    model = _TinyTwinModel()
    loader = DataLoader(_TinyVariantDataset(), batch_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = _RecordingGradScaler(events)

    def record_clip(parameters, max_norm):
        list(parameters)
        events.append(("clip", max_norm))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)

    _run_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        amp_dtype=torch.float16,
        grad_scaler=scaler,
    )

    assert events == [
        "scale",
        "backward",
        "unscale",
        ("clip", 1.0),
        "step",
        "update",
    ] * len(loader)


def test_finetune_clips_each_parameter_independently(monkeypatch):
    calls = []
    model = _TwoParameterTwinModel()
    loader = DataLoader(_TinyVariantDataset(), batch_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def record_clip(parameters, max_norm):
        calls.append((list(parameters), max_norm))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)

    _run_epoch(model, loader, optimizer, torch.device("cpu"), max_steps=1)

    assert len(calls) == 2
    assert all(len(parameters) == 1 for parameters, _ in calls)
    assert all(max_norm == 1.0 for _, max_norm in calls)


def test_twin_model_keeps_frozen_batchnorm_in_eval_mode():
    base_model = PromoterAI(
        num_blocks=4, model_dim=8, output_dims=[3], output_crop=0
    )
    twin_model = TwinModel(base_model)
    before = base_model.blocks[0].bn1.running_mean.detach().clone()
    loader = DataLoader(_TinyVariantDataset(), batch_size=2)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, twin_model.parameters()), lr=1e-3
    )

    _run_epoch(twin_model, loader, optimizer, torch.device("cpu"), max_steps=1)

    assert base_model.blocks[0].bn1.training is False
    assert torch.equal(base_model.blocks[0].bn1.running_mean, before)


def test_autocast_is_disabled_on_cpu(monkeypatch):
    def unexpected_autocast(*args, **kwargs):
        raise AssertionError("torch.autocast should not be called for CPU finetuning")

    monkeypatch.setattr(torch, "autocast", unexpected_autocast)

    with autocast_context(torch.device("cpu"), torch.bfloat16):
        pass


@pytest.mark.parametrize("amp_dtype", ["none", "bf16", "fp16"])
def test_unified_finetune_cli_accepts_amp_choices(amp_dtype):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from promoterai_torch.cli import main; "
                f"sys.argv=['promoterai-torch', 'finetune', '--amp_dtype', '{amp_dtype}', '--help']; "
                "main()"
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--amp_dtype {none,bf16,fp16}" in result.stdout


@pytest.mark.parametrize(
    "command",
    [
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from promoterai_torch.cli import main; "
                "sys.argv=['promoterai-torch', 'finetune', '--help']; "
                "main()"
            ),
        ],
        [sys.executable, "-m", "promoterai_torch.finetune", "--help"],
    ],
)
def test_finetune_clis_expose_resume_options(command):
    result = subprocess.run(command, capture_output=True, text=True)

    assert result.returncode == 0
    assert "--resume_checkpoint" in result.stdout
    assert "--auto_resume" in result.stdout


def test_resolve_finetune_resume_checkpoint_prefers_explicit(tmp_path):
    latest = tmp_path / "latest_model.pt"
    latest.write_text("placeholder")
    explicit = tmp_path / "custom.pt"

    resolved = resolve_finetune_resume_checkpoint(
        str(explicit), auto_resume=True, checkpoint_folder=str(tmp_path)
    )

    assert resolved == str(explicit)


def test_resolve_finetune_resume_checkpoint_uses_latest_when_present(tmp_path):
    latest = tmp_path / "latest_model.pt"
    latest.write_text("placeholder")

    resolved = resolve_finetune_resume_checkpoint(
        None, auto_resume=True, checkpoint_folder=str(tmp_path)
    )

    assert resolved == str(latest)


def test_resolve_finetune_resume_checkpoint_starts_fresh_without_latest(tmp_path):
    assert (
        resolve_finetune_resume_checkpoint(
            None, auto_resume=True, checkpoint_folder=str(tmp_path)
        )
        is None
    )
    assert (
        resolve_finetune_resume_checkpoint(
            None, auto_resume=False, checkpoint_folder=str(tmp_path)
        )
        is None
    )


def test_resolve_finetune_epoch_sizes_matches_official_complete_batches():
    steps, val_samples = resolve_finetune_epoch_sizes(
        train_size=103, val_size=23, global_batch_size=8
    )

    assert steps == (103 // 8) // 5
    assert val_samples == (23 // 8) * 8


@pytest.mark.parametrize(("train_size", "val_size"), [(39, 8), (40, 7)])
def test_resolve_finetune_epoch_sizes_rejects_insufficient_batches(
    train_size, val_size
):
    with pytest.raises(ValueError):
        resolve_finetune_epoch_sizes(train_size, val_size, global_batch_size=8)


def test_distributed_slice_sampler_has_no_padding_or_duplicates():
    rank_indices = [
        list(DistributedSliceSampler(total_size=16, rank=rank, world_size=4))
        for rank in range(4)
    ]

    assert all(len(indices) == 4 for indices in rank_indices)
    assert sorted(index for indices in rank_indices for index in indices) == list(
        range(16)
    )


def test_finetune_optimizer_matches_keras_epsilon():
    model = _TinyTwinModel()

    optimizer = build_finetune_optimizer(model, 1e-3, 5e-6)

    assert optimizer.defaults["eps"] == 1e-7
    assert optimizer.defaults["weight_decay"] == 5e-6


def test_finetune_checkpoint_is_load_pretrained_compatible(tmp_path):
    base_model = PromoterAI(
        num_blocks=4,
        model_dim=8,
        output_dims=[3],
        output_crop=4,
    )
    twin_model = TwinModel(base_model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, twin_model.parameters()), lr=1e-3
    )
    base_args = {
        "num_blocks": 4,
        "model_dim": 8,
        "output_dims": [3],
        "input_length": 12,
        "output_length": 8,
        "output_crop": 4,
        "shortcut_layer_freq": 4,
    }
    finetune_args = {"batch_size": 2, "epochs": 1}

    save_finetune_checkpoint(
        twin_model,
        optimizer,
        val_loss=0.25,
        epoch=3,
        checkpoint_folder=str(tmp_path),
        base_args=base_args,
        finetune_args=finetune_args,
        inference_only=True,
    )

    loaded_model, args = load_pretrained(str(tmp_path / "best_model.pt"))
    checkpoint = torch.load(tmp_path / "best_model.pt", map_location="cpu")

    assert args["num_blocks"] == 4
    assert args["model_dim"] == 8
    assert args["output_dims"] == [3]
    assert args["output_crop"] == 4
    assert args["finetune_args"] == finetune_args
    assert set(checkpoint) == {"model_state_dict", "args"}
    for key, value in twin_model.base_model.state_dict().items():
        assert torch.equal(value, loaded_model.state_dict()[key])


def test_finetune_checkpoint_restores_training_state(tmp_path):
    twin_model, optimizer, base_args = _build_twin_model_optimizer()
    scaler = _StatefulGradScaler({"scale": 128.0})
    first_param = next(twin_model.base_model.parameters())
    first_param.data.fill_(0.25)
    optimizer.param_groups[0]["lr"] = 2e-4
    torch.manual_seed(1234)

    save_finetune_checkpoint(
        twin_model,
        optimizer,
        val_loss=0.4,
        epoch=6,
        checkpoint_folder=str(tmp_path),
        base_args=base_args,
        finetune_args={"epochs": 10, "amp_dtype": "fp16"},
        checkpoint_name="latest_model.pt",
        best_val_loss=0.2,
        grad_scaler=scaler,
        atomic=True,
    )
    expected_random = torch.rand(4)
    assert not list(tmp_path.glob(".latest_model.pt.*.tmp"))

    fresh_model, fresh_optimizer, _ = _build_twin_model_optimizer()
    fresh_scaler = _StatefulGradScaler({"scale": 1.0})
    torch.manual_seed(999)
    start_epoch, best_val_loss, args = load_finetune_checkpoint(
        fresh_model,
        fresh_optimizer,
        fresh_scaler,
        str(tmp_path / "latest_model.pt"),
        torch.device("cpu"),
    )

    assert start_epoch == 7
    assert best_val_loss == 0.2
    assert args["finetune_args"]["amp_dtype"] == "fp16"
    assert fresh_optimizer.param_groups[0]["lr"] == 2e-4
    assert fresh_scaler.state_dict() == {"scale": 128.0}
    assert torch.equal(next(fresh_model.base_model.parameters()), first_param)
    assert torch.equal(torch.rand(4), expected_random)


def test_finetune_checkpoint_helpers_unwrap_distributed_model(tmp_path):
    twin_model, optimizer, base_args = _build_twin_model_optimizer()
    wrapped = _ModuleWrapper(twin_model)

    save_finetune_checkpoint(
        wrapped,
        optimizer,
        val_loss=0.4,
        epoch=1,
        checkpoint_folder=str(tmp_path),
        base_args=base_args,
        finetune_args={"epochs": 2},
        checkpoint_name="latest_model.pt",
        best_val_loss=0.3,
    )
    fresh_model, fresh_optimizer, _ = _build_twin_model_optimizer()
    fresh_wrapped = _ModuleWrapper(fresh_model)

    start_epoch, best_val_loss, _ = load_finetune_checkpoint(
        fresh_wrapped,
        fresh_optimizer,
        grad_scaler=None,
        checkpoint_path=str(tmp_path / "latest_model.pt"),
        device=torch.device("cpu"),
    )

    assert start_epoch == 2
    assert best_val_loss == 0.3
    for key, value in twin_model.base_model.state_dict().items():
        assert torch.equal(value, fresh_model.base_model.state_dict()[key])


def test_load_finetune_checkpoint_accepts_legacy_checkpoint(tmp_path):
    twin_model, optimizer, base_args = _build_twin_model_optimizer()
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": twin_model.base_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": 0.35,
            "epoch": 2,
            "args": base_args,
        },
        checkpoint_path,
    )
    fresh_model, fresh_optimizer, _ = _build_twin_model_optimizer()
    scaler = _StatefulGradScaler({"scale": 16.0})

    start_epoch, best_val_loss, loaded_args = load_finetune_checkpoint(
        fresh_model,
        fresh_optimizer,
        scaler,
        str(checkpoint_path),
        torch.device("cpu"),
    )

    assert start_epoch == 3
    assert best_val_loss == 0.35
    assert loaded_args == base_args
    assert scaler.state_dict() == {"scale": 16.0}


def _build_twin_model_optimizer():
    base_model = PromoterAI(
        num_blocks=4,
        model_dim=8,
        output_dims=[3],
        output_crop=4,
    )
    twin_model = TwinModel(base_model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, twin_model.parameters()), lr=1e-3
    )
    base_args = {
        "num_blocks": 4,
        "model_dim": 8,
        "output_dims": [3],
        "input_length": 12,
        "output_length": 8,
        "output_crop": 4,
        "shortcut_layer_freq": 4,
    }
    return twin_model, optimizer, base_args


class _TinyVariantDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        x_ref = torch.ones(4, 4)
        x_alt = torch.full((4, 4), 2.0)
        return (x_ref, x_alt), torch.tensor(0.0)


def _variant_row(
    gene,
    *,
    in_cds=0,
    spliceai=0.01,
    p_under=1.0,
    p_over=1.0,
):
    return {
        "gene": gene,
        "in_cds": in_cds,
        "spliceai": spliceai,
        "p_under": p_under,
        "p_over": p_over,
    }


class _TinyTwinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_ref, x_alt):
        return self.scale * (x_alt - x_ref).mean(dim=(1, 2))


class _TwoParameterTwinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_ref, x_alt):
        return self.scale * (x_alt - x_ref).mean(dim=(1, 2)) + self.bias


class _ModuleWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


class _ScaledLoss:
    def __init__(self, loss, events):
        self.loss = loss
        self.events = events

    def backward(self):
        self.events.append("backward")
        self.loss.backward()


class _RecordingGradScaler:
    def __init__(self, events):
        self.events = events

    def is_enabled(self):
        return True

    def scale(self, loss):
        self.events.append("scale")
        return _ScaledLoss(loss, self.events)

    def unscale_(self, optimizer):
        self.events.append("unscale")

    def step(self, optimizer):
        self.events.append("step")
        optimizer.step()

    def update(self):
        self.events.append("update")


class _StatefulGradScaler:
    def __init__(self, state):
        self.state = dict(state)

    def is_enabled(self):
        return True

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state):
        self.state = dict(state)
