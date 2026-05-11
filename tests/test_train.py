import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from promoterai_torch.architecture import PromoterAI
from promoterai_torch.train import _run_epoch, compute_loss, load_training_checkpoint
from promoterai_torch.utils import save_checkpoint


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


class _FakeWandbRun:
    def __init__(self):
        self.logged = []

    def log(self, metrics, step=None):
        self.logged.append((metrics, step))
