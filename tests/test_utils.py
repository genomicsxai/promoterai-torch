import sys
import types

import pytest
import torch

from promoterai_torch.utils import (
    apply_optimizer_schedule,
    export_inference_checkpoint,
    finish_wandb,
    init_wandb,
    log_wandb,
    make_lr_lambda,
    normalize_model_state_dict,
    save_checkpoint,
    setup_distributed,
    unwrap_model,
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


def test_unwrap_model_handles_torch_compile_wrappers():
    model = torch.nn.Linear(2, 3)
    compiled = torch.compile(model)

    assert unwrap_model(compiled) is model


def test_save_checkpoint_strips_torch_compile_state_prefix(tmp_path):
    model = torch.nn.Linear(2, 3)
    compiled = torch.compile(model)
    optimizer = torch.optim.AdamW(compiled.parameters(), lr=1e-3)

    save_checkpoint(
        compiled,
        optimizer,
        scheduler=None,
        val_loss=0.5,
        epoch=2,
        checkpoint_folder=str(tmp_path),
        args_dict={"kind": "linear"},
        checkpoint_name="latest_model.pt",
    )

    ckpt = torch.load(str(tmp_path / "latest_model.pt"), map_location="cpu")

    assert set(ckpt["model_state_dict"]) == {"weight", "bias"}


def test_save_checkpoint_can_write_inference_only_checkpoint(tmp_path):
    model = torch.nn.Linear(2, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    save_checkpoint(
        model,
        optimizer,
        scheduler=None,
        val_loss=0.5,
        epoch=2,
        checkpoint_folder=str(tmp_path),
        args_dict={"kind": "linear"},
        checkpoint_name="best_model.pt",
        inference_only=True,
    )

    checkpoint = torch.load(tmp_path / "best_model.pt", map_location="cpu")

    assert set(checkpoint) == {"model_state_dict", "args"}


def test_export_inference_checkpoint_strips_training_state(tmp_path):
    source = tmp_path / "latest_model.pt"
    output = tmp_path / "exports" / "model.pt"
    torch.save(
        {
            "model_state_dict": {
                "module._orig_mod.weight": torch.tensor([1.0]),
            },
            "optimizer_state_dict": {"state": {"large": torch.ones(100)}},
            "epoch": 7,
            "args": {"num_blocks": 4},
        },
        source,
    )

    export_inference_checkpoint(source, output)
    checkpoint = torch.load(output, map_location="cpu")

    assert set(checkpoint) == {"model_state_dict", "args"}
    assert set(checkpoint["model_state_dict"]) == {"weight"}
    assert checkpoint["args"] == {"num_blocks": 4}
    assert output.stat().st_size < source.stat().st_size


def test_export_inference_checkpoint_rejects_in_place_conversion(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": {}, "args": {}}, checkpoint)

    with pytest.raises(ValueError, match="must be different"):
        export_inference_checkpoint(checkpoint, checkpoint)


def test_export_inference_cli(tmp_path, monkeypatch):
    from promoterai_torch.cli import main

    source = tmp_path / "latest_model.pt"
    output = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "optimizer_state_dict": {"state": {}},
            "args": {"num_blocks": 4},
        },
        source,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promoterai-torch",
            "export-inference",
            "--checkpoint",
            str(source),
            "--output",
            str(output),
        ],
    )

    main()

    assert set(torch.load(output, map_location="cpu")) == {
        "model_state_dict",
        "args",
    }


def test_normalize_model_state_dict_strips_wrapper_prefixes():
    state = {
        "_orig_mod.weight": torch.tensor([1.0]),
        "module._orig_mod.bias": torch.tensor([2.0]),
    }

    normalized = normalize_model_state_dict(state)

    assert set(normalized) == {"weight", "bias"}
    assert torch.equal(normalized["weight"], state["_orig_mod.weight"])
    assert torch.equal(normalized["bias"], state["module._orig_mod.bias"])


def test_setup_distributed_returns_global_rank_for_multinode(monkeypatch):
    calls = {}

    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def fake_init_process_group(backend):
        calls["backend"] = backend

    monkeypatch.setattr(
        "promoterai_torch.utils.dist.init_process_group", fake_init_process_group
    )
    monkeypatch.setattr("promoterai_torch.utils.dist.get_rank", lambda: 5)
    monkeypatch.setattr("promoterai_torch.utils.dist.get_world_size", lambda: 8)

    rank, world_size, device = setup_distributed()

    assert calls == {"backend": "gloo"}
    assert rank == 5
    assert world_size == 8
    assert device == torch.device("cpu")


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
