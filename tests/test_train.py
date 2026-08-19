import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from promoterai_torch.architecture import PromoterAI
from promoterai_torch.train import (
    _run_epoch,
    build_train_optimizer,
    compute_loss,
    load_training_checkpoint,
    resolve_amp_dtype,
    resolve_per_rank_batch_size,
    resolve_resume_checkpoint,
    resolve_train_steps_per_epoch,
)
from promoterai_torch.utils import save_checkpoint


def test_resolve_per_rank_batch_size_keeps_batch_global():
    assert resolve_per_rank_batch_size(global_batch_size=32, world_size=8) == 4
    assert resolve_per_rank_batch_size(global_batch_size=32, world_size=1) == 32


def test_resolve_per_rank_batch_size_rejects_non_divisible_global_batch():
    try:
        resolve_per_rank_batch_size(global_batch_size=32, world_size=6)
    except ValueError as exc:
        assert "global batch size" in str(exc)
        assert "divisible" in str(exc)
    else:
        raise AssertionError("Expected non-divisible global batch size to fail")


def test_resolve_train_steps_per_epoch_divides_out_batch_size():
    # Illumina's train.py counts pre-batched tf.data elements (i.e. batches,
    # not raw samples) before dividing by 10, so dataset_sizes (raw sample
    # counts) must be divided by the global batch size first to match —
    # otherwise steps_per_epoch (and thus samples trained per epoch) comes
    # out global_batch_size times too large.
    assert resolve_train_steps_per_epoch([10000], global_batch_size=10) == 100
    assert resolve_train_steps_per_epoch([7000, 3000], global_batch_size=10) == 100


def test_build_train_optimizer_matches_keras_epsilon_default():
    model = nn.Linear(2, 2)
    optimizer = build_train_optimizer(model, learning_rate=5e-4, weight_decay=5e-6)
    assert optimizer.defaults["eps"] == 1e-7
    assert optimizer.defaults["lr"] == 5e-4
    assert optimizer.defaults["weight_decay"] == 5e-6


def test_resolve_amp_dtype_maps_cli_values():
    assert resolve_amp_dtype("none") is None
    assert resolve_amp_dtype("bf16") is torch.bfloat16
    assert resolve_amp_dtype("fp16") is torch.float16


def test_resolve_resume_checkpoint_prefers_explicit_checkpoint(tmp_path):
    explicit = tmp_path / "custom.pt"
    latest = tmp_path / "latest_model.pt"
    latest.write_text("placeholder")
    args = _Args(
        checkpoint_folder=str(tmp_path),
        resume_checkpoint=str(explicit),
        auto_resume=True,
    )

    assert resolve_resume_checkpoint(args) == str(explicit)


def test_resolve_resume_checkpoint_uses_latest_when_auto_resume_enabled(tmp_path):
    latest = tmp_path / "latest_model.pt"
    latest.write_text("placeholder")
    args = _Args(
        checkpoint_folder=str(tmp_path),
        resume_checkpoint=None,
        auto_resume=True,
    )

    assert resolve_resume_checkpoint(args) == str(latest)


def test_resolve_resume_checkpoint_ignores_latest_without_auto_resume(tmp_path):
    latest = tmp_path / "latest_model.pt"
    latest.write_text("placeholder")
    args = _Args(
        checkpoint_folder=str(tmp_path),
        resume_checkpoint=None,
        auto_resume=False,
    )

    assert resolve_resume_checkpoint(args) is None


def test_load_training_checkpoint_restores_model_optimizer_and_epoch(tmp_path):
    model = PromoterAI(num_blocks=4, model_dim=8, output_dims=[3], output_crop=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.zeros(1, 8, 4)
    x[:, :, 0] = 1.0
    loss = model(x)[0].sum()
    loss.backward()
    optimizer.step()
    expected_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    args = {
        "num_blocks": 4,
        "model_dim": 8,
        "output_dims": [3],
        "input_length": 8,
        "output_length": 8,
        "output_crop": 0,
    }

    save_checkpoint(
        model,
        optimizer,
        scheduler=None,
        val_loss=0.4,
        epoch=6,
        checkpoint_folder=str(tmp_path),
        args_dict=args,
        checkpoint_name="latest_model.pt",
        best_val_loss=0.25,
    )
    ckpt = torch.load(str(tmp_path / "latest_model.pt"), map_location="cpu")
    assert ckpt["epoch"] == 6
    assert ckpt["optimizer_state_dict"] is not None
    assert ckpt["optimizer_state_dict"]["state"]
    assert "scheduler_state_dict" in ckpt
    assert ckpt["scheduler_state_dict"] is None

    fresh_model = PromoterAI(num_blocks=4, model_dim=8, output_dims=[3], output_crop=0)
    fresh_optimizer = torch.optim.AdamW(
        fresh_model.parameters(), lr=9e-3, weight_decay=9e-4
    )
    start_epoch, best_val_loss, loaded_args = load_training_checkpoint(
        fresh_model,
        fresh_optimizer,
        str(tmp_path / "latest_model.pt"),
        torch.device("cpu"),
    )

    assert start_epoch == 7
    assert best_val_loss == 0.25
    assert loaded_args == args
    assert fresh_optimizer.param_groups[0]["lr"] == 1e-3
    assert fresh_optimizer.param_groups[0]["weight_decay"] == 1e-4
    assert fresh_optimizer.state_dict()["state"]
    for key, value in fresh_model.state_dict().items():
        assert torch.equal(value, expected_state[key])


def test_load_training_checkpoint_accepts_compiled_state_prefix(tmp_path):
    model = PromoterAI(num_blocks=4, model_dim=8, output_dims=[3], output_crop=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    prefixed_state = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    torch.save(
        {
            "model_state_dict": prefixed_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 2,
            "best_val_loss": 0.7,
            "args": {},
        },
        str(tmp_path / "compiled.pt"),
    )

    fresh_model = PromoterAI(num_blocks=4, model_dim=8, output_dims=[3], output_crop=0)
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=9e-3)
    start_epoch, best_val_loss, _ = load_training_checkpoint(
        fresh_model,
        fresh_optimizer,
        str(tmp_path / "compiled.pt"),
        torch.device("cpu"),
    )

    assert start_epoch == 3
    assert best_val_loss == 0.7
    for key, value in fresh_model.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])


def test_compute_loss_handles_batch_size_one_weights():
    outputs = (
        torch.ones(1, 4, 2),
        torch.full((1, 4, 3), 2.0),
    )
    y_tuple = (
        torch.zeros(1, 4, 2),
        torch.zeros(1, 4, 3),
    )
    w_tuple = torch.tensor([[1.0, 0.5]])

    loss = compute_loss(outputs, y_tuple, w_tuple)

    expected = F.mse_loss(outputs[0], y_tuple[0]) + 0.5 * F.mse_loss(
        outputs[1], y_tuple[1]
    )
    assert loss == expected


def test_compute_loss_accepts_uncollated_single_sample_weights():
    outputs = (torch.ones(1, 4, 2),)
    y_tuple = (torch.zeros(1, 4, 2),)
    w_tuple = torch.tensor([0.25])

    loss = compute_loss(outputs, y_tuple, w_tuple)

    assert loss == 0.25 * F.mse_loss(outputs[0], y_tuple[0])


def test_run_epoch_clips_each_parameter_independently(monkeypatch):
    calls = []
    model = _TwoParameterTrackModel()
    loader = DataLoader(_TinyTrackDataset(), batch_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def record_clip(parameters, max_norm):
        calls.append((list(parameters), max_norm))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)

    _run_epoch(model, loader, optimizer, torch.device("cpu"), max_steps=1)

    # Keras' AdamW(clipnorm=...) (used by Illumina's from-scratch training)
    # clips each variable's gradient norm independently rather than clipping
    # the norm across all parameters jointly, so each parameter must get its
    # own clip_grad_norm_ call.
    assert len(calls) == 2
    assert all(len(parameters) == 1 for parameters, _ in calls)
    assert all(max_norm == 1e-4 for _, max_norm in calls)


def test_run_epoch_supports_progress_and_batch_logging(capsys):
    model = _TinyTrackModel()
    loader = DataLoader(_TinyTrackDataset(), batch_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss = _run_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        desc="train 1/1",
        show_progress=False,
        log_every_batches=1,
    )

    captured = capsys.readouterr()
    assert "train 1/1 batch 1" in captured.out
    assert loss > 0.0


def test_run_epoch_returns_profile_metrics():
    model = _TinyTrackModel()
    loader = DataLoader(_TinyTrackDataset(), batch_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss, metrics = _run_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        profile_batches=1,
        profile_warmup_batches=0,
        return_metrics=True,
    )

    assert loss > 0.0
    assert metrics["global_samples"] == 1.0
    assert metrics["profile_samples"] == 1.0
    assert metrics["samples_per_sec"] > 0.0
    assert metrics["profile_samples_per_sec"] > 0.0


def test_run_epoch_logs_batch_loss_and_lr_to_wandb():
    model = _TinyTrackModel()
    loader = DataLoader(_TinyTrackDataset(), batch_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    run = _FakeWandbRun()

    _run_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        desc="train 1/1",
        wandb_run=run,
        wandb_prefix="train",
        wandb_step_offset=10,
        wandb_log_every_batches=1,
    )

    assert len(run.logged) == 2
    metrics, step = run.logged[0]
    assert step == 11
    assert metrics["train/batch_loss"] > 0.0
    assert metrics["train/running_loss"] > 0.0
    assert metrics["optim/lr"] == 1e-3
    assert metrics["optim/weight_decay"] == 1e-4


class _TinyTrackDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        x = torch.ones(4, 4)
        y = torch.zeros(4, 2)
        w = torch.tensor([1.0])
        return x, (y,), w


class _TinyTrackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        out = self.scale * torch.ones(x.shape[0], x.shape[1], 2, device=x.device)
        return (out,)


class _TwoParameterTrackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        ones = torch.ones(x.shape[0], x.shape[1], 2, device=x.device)
        out = self.scale * ones + self.bias * ones
        return (out,)


class _FakeWandbRun:
    def __init__(self):
        self.logged = []

    def log(self, metrics, step=None):
        self.logged.append((metrics, step))


class _Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
