import pytest
import torch
import sys
import types

from promoterai_torch.utils import (
    apply_optimizer_schedule,
    finish_wandb,
    init_wandb,
    log_wandb,
    make_lr_lambda,
)


def test_lr_schedule_warmup():
    fn = make_lr_lambda(100)
    # Linear warmup: epoch 0 → 1/10=0.1, epoch 9 → 10/10=1.0
    assert abs(fn(0) - 0.1) < 1e-9
    assert abs(fn(9) - 1.0) < 1e-9


def test_lr_schedule_constant():
    fn = make_lr_lambda(100)
    for epoch in [10, 50, 89]:
        assert fn(epoch) == 1.0, f"epoch {epoch} should be 1.0, got {fn(epoch)}"


def test_lr_schedule_decay():
    fn = make_lr_lambda(100)
    # epoch 90: (100-90)/(0.1*100) = 1.0
    # epoch 91: (100-91)/10 = 0.9
    # epoch 99: (100-99)/10 = 0.1
    assert abs(fn(90) - 1.0) < 1e-9
    assert abs(fn(91) - 0.9) < 1e-9
    assert abs(fn(99) - 0.1) < 1e-9


def test_lr_schedule_boundary():
    fn = make_lr_lambda(10)
    # total=10: warmup epoch<1 (epoch 0 only), decay epoch>9 (none for 0-indexed 0..9)
    # epoch 0: warmup → (0+1)/(0.1*10) = 1.0
    assert fn(0) == pytest.approx(1.0)
    # epoch 9: 9 > 0.9*10=9.0 → False → constant region → 1.0
    assert fn(9) == pytest.approx(1.0)


def test_lr_lambda_non_negative():
    fn = make_lr_lambda(200)
    for epoch in range(200):
        assert fn(epoch) >= 0


def test_apply_optimizer_schedule_updates_lr_and_weight_decay_at_epoch_begin():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([param], lr=5e-4, weight_decay=5e-6)

    scale = apply_optimizer_schedule(
        optimizer,
        base_lr=5e-4,
        base_wd=5e-6,
        total_epochs=100,
        epoch=0,
    )

    assert scale == pytest.approx(0.1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(5e-7)


def test_init_wandb_noops_without_project_or_on_nonzero_rank():
    args = types.SimpleNamespace(wandb_project=None, wandb_mode=None)

    assert init_wandb(args, {"x": 1}, rank=0) is None
    args.wandb_project = "project"
    assert init_wandb(args, {"x": 1}, rank=1) is None
    args.wandb_mode = "disabled"
    assert init_wandb(args, {"x": 1}, rank=0) is None


def test_init_wandb_imports_optional_dependency_and_forwards_config(monkeypatch):
    calls = {}

    class FakeRun:
        pass

    def fake_init(**kwargs):
        calls.update(kwargs)
        return FakeRun()

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=fake_init))
    args = types.SimpleNamespace(
        wandb_project="promoterai",
        wandb_entity="lab",
        wandb_run_name="run-1",
        wandb_mode="offline",
        wandb_tags=["train", "smoke"],
    )
    config = {"epochs": 3}

    run = init_wandb(args, config, rank=0)

    assert isinstance(run, FakeRun)
    assert calls == {
        "project": "promoterai",
        "entity": "lab",
        "name": "run-1",
        "mode": "offline",
        "tags": ["train", "smoke"],
        "config": config,
    }


def test_wandb_log_and_finish_delegate_to_run():
    class FakeRun:
        def __init__(self):
            self.logged = []
            self.finished = False

        def log(self, metrics, step=None):
            self.logged.append((metrics, step))

        def finish(self):
            self.finished = True

    run = FakeRun()

    log_wandb(run, {"loss": 0.5}, step=2)
    finish_wandb(run)
    log_wandb(None, {"ignored": True}, step=3)
    finish_wandb(None)

    assert run.logged == [({"loss": 0.5}, 2)]
    assert run.finished
