"""
Tests for the Keras → PyTorch weight converter.

Skipped automatically when tf-keras is not installed.
Models are built using tf_keras subclassed layers to match the weight naming
convention of the Illumina SavedModel (Dense stem, MetaFormerBlock sublayers,
shortcut_{species}{N} output projections).
"""

import numpy as np
import pytest
import torch

pytest.importorskip("tf_keras", reason="tf-keras not installed")


def _build_tf_keras_model(
    num_blocks=8,
    model_dim=16,
    output_dims=(8,),
    output_crop=0,
    kernel_size=5,
    shortcut_layer_freq=4,
    species=("human",),
):
    """Build a minimal PromoterAI-like model matching the Illumina subclassed weight format."""
    import tf_keras as keras

    class MetaFormerBlock(keras.layers.Layer):
        def __init__(self, model_dim, kernel_size, dilation_rate, **kwargs):
            super().__init__(**kwargs)
            # Explicit names ensure weight paths match the real model regardless of
            # the global Keras name counter (which carries over between test runs).
            self.depthwise_conv1d = keras.layers.DepthwiseConv1D(
                kernel_size,
                dilation_rate=dilation_rate,
                padding="same",
                use_bias=True,
                name="depthwise_conv1d",
            )
            self.batch_normalization = keras.layers.BatchNormalization(
                name="batch_normalization"
            )
            self.batch_normalization_1 = keras.layers.BatchNormalization(
                name="batch_normalization_1"
            )
            self.dense = keras.layers.Dense(
                model_dim * 4, activation="relu", name="dense"
            )
            self.dense_1 = keras.layers.Dense(model_dim, name="dense_1")

        def call(self, x, training=None):
            x_norm = self.batch_normalization(x, training=training)
            intermediate = x + self.depthwise_conv1d(x_norm)
            x_norm2 = self.batch_normalization_1(intermediate, training=training)
            return intermediate + self.dense_1(self.dense(x_norm2))

    inp = keras.Input(shape=(None, 4))
    # Stem: explicitly named 'dense' to match the real model's weight path
    x = keras.layers.Dense(model_dim, activation="relu", name="dense")(inp)

    # Blocks: named meta_former_block, meta_former_block_1, ...
    block_outputs = [None] * (num_blocks + 1)
    block_outputs[0] = x
    for i in range(num_blocks):
        dilation = max(1, 2 ** (i // 2 - 1))
        name = "meta_former_block" if i == 0 else f"meta_former_block_{i}"
        block_outputs[i + 1] = MetaFormerBlock(
            model_dim, kernel_size, dilation, name=name
        )(block_outputs[i])

    # Output heads: shortcut_{species}{block_num} naming
    shortcut_nums = list(range(num_blocks, 0, -shortcut_layer_freq))
    outputs = []
    for sp, od in zip(species, output_dims):
        projs = [
            keras.layers.Dense(od, activation="relu", name=f"shortcut_{sp}{n}")(
                block_outputs[n]
            )
            for n in shortcut_nums
        ]
        head = keras.layers.Average()(projs) if len(projs) > 1 else projs[0]
        if output_crop > 0:
            head = keras.layers.Cropping1D(output_crop // 2)(head)
        outputs.append(head)

    return keras.Model(inputs=inp, outputs=outputs if len(outputs) > 1 else outputs[0])


def _save_savedmodel(model, path: str):
    """Save a tf_keras model in SavedModel format (directory, no extension)."""
    import tf_keras as keras

    keras.models.save_model(model, path, save_format="tf")


def test_convert_infers_architecture(tmp_path):
    """Converter correctly infers num_blocks, model_dim, output_dims from the Keras model."""
    from promoterai_torch.utils import convert_tf_weights

    model = _build_tf_keras_model(num_blocks=8, model_dim=16, output_dims=(8,))
    _save_savedmodel(model, str(tmp_path / "keras_model"))

    out_pt = str(tmp_path / "model.pt")
    convert_tf_weights(str(tmp_path / "keras_model"), out_pt)

    ckpt = torch.load(out_pt, map_location="cpu")
    assert ckpt["args"]["num_blocks"] == 8
    assert ckpt["args"]["model_dim"] == 16
    assert ckpt["args"]["output_dims"] == [8]


def test_convert_loads_cleanly(tmp_path):
    """Converted checkpoint loads into PromoterAI without missing keys."""
    from promoterai_torch.utils import convert_tf_weights, load_pretrained

    model = _build_tf_keras_model(num_blocks=8, model_dim=16, output_dims=(8,))
    _save_savedmodel(model, str(tmp_path / "keras_model"))

    out_pt = str(tmp_path / "model.pt")
    convert_tf_weights(
        str(tmp_path / "keras_model"), out_pt, input_length=512, output_length=512
    )

    pt_model, args = load_pretrained(out_pt)
    assert args["input_length"] == 512
    pt_model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 64, 4)
        x[:, :, 0] = 1.0
        out = pt_model(x)
    assert len(out) == 1


def test_convert_multi_species(tmp_path):
    """Converter handles multiple output heads (e.g. human + mouse)."""
    from promoterai_torch.utils import convert_tf_weights, load_pretrained

    model = _build_tf_keras_model(
        num_blocks=8, model_dim=16, output_dims=(8, 6), species=("human", "mouse")
    )
    _save_savedmodel(model, str(tmp_path / "keras_model"))

    out_pt = str(tmp_path / "model.pt")
    convert_tf_weights(str(tmp_path / "keras_model"), out_pt)

    pt_model, args = load_pretrained(out_pt)
    assert args["output_dims"] == [8, 6]
    pt_model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 64, 4)
        out = pt_model(x)
    assert len(out) == 2


def test_convert_numerical_parity(tmp_path):
    """PyTorch and Keras models produce numerically matching outputs after weight conversion."""
    from promoterai_torch.utils import convert_tf_weights, load_pretrained

    input_len = 64
    keras_model = _build_tf_keras_model(num_blocks=8, model_dim=16, output_dims=(8,))
    _save_savedmodel(keras_model, str(tmp_path / "keras_model"))

    out_pt = str(tmp_path / "model.pt")
    convert_tf_weights(str(tmp_path / "keras_model"), out_pt)
    pt_model, _ = load_pretrained(out_pt)
    pt_model.eval()

    rng = np.random.default_rng(0)
    indices = rng.integers(0, 4, size=input_len)
    x_np = np.eye(4, dtype="float32")[indices][None]  # (1, L, 4)

    keras_out = keras_model(x_np, training=False)
    if isinstance(keras_out, (list, tuple)):
        keras_out = keras_out[0]
    keras_out = np.array(keras_out)  # (1, L, 8)

    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        pt_out = pt_model(x_pt)[0].numpy()  # (1, L, 8)

    np.testing.assert_allclose(
        pt_out,
        keras_out,
        atol=1e-4,
        err_msg="Keras and PyTorch outputs differ after conversion",
    )
