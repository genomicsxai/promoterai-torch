import torch

from promoterai_torch.architecture import PromoterAI, TwinModel
from promoterai_torch.finetune import save_finetune_checkpoint
from promoterai_torch.utils import load_pretrained


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
    )

    loaded_model, args = load_pretrained(str(tmp_path / "best_model.pt"))
    checkpoint = torch.load(tmp_path / "best_model.pt", map_location="cpu")

    assert args["num_blocks"] == 4
    assert args["model_dim"] == 8
    assert args["output_dims"] == [3]
    assert args["output_crop"] == 4
    assert args["finetune_args"] == finetune_args
    assert checkpoint["val_loss"] == 0.25
    assert checkpoint["epoch"] == 3
    assert checkpoint["optimizer_state_dict"] is not None
    assert checkpoint["scheduler_state_dict"] is None
    for key, value in twin_model.base_model.state_dict().items():
        assert torch.equal(value, loaded_model.state_dict()[key])
