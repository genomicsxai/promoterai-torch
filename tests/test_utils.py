import pytest
import torch

from promoterai_torch.utils import apply_optimizer_schedule, make_lr_lambda


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
