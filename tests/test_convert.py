"""
Tests for the Keras → PyTorch weight converter.

Skipped automatically when tf-keras is not installed.
Models are built and saved using tf_keras (Keras 2 API) so they match the
SavedModel format that Illumina distributes.
"""
import pytest
import torch
import numpy as np

pytest.importorskip("tf_keras", reason="tf-keras not installed")


def _build_tf_keras_model(num_blocks=8, model_dim=16, output_dims=(8,), output_crop=0,
                          kernel_size=5, shortcut_layer_freq=4):
    """Build a minimal PromoterAI-like model using tf_keras (Keras 2 API)."""
    import tf_keras as keras
    from functools import partial

    _kernel_init = partial(keras.initializers.TruncatedNormal, stddev=0.01)

    def _metaformer(md, ks, dr):
        def block(inp):
            x = keras.layers.BatchNormalization(synchronized=True)(inp)
            x = keras.layers.DepthwiseConv1D(ks, dilation_rate=dr, padding='same')(x)
            mid = inp + x
            x = keras.layers.BatchNormalization(synchronized=True)(mid)
            x = keras.layers.Dense(md * 4, activation='relu',
                                   kernel_initializer=_kernel_init())(x)
            x = keras.layers.Dense(md, kernel_initializer=_kernel_init())(x)
            return mid + x
        return block

    inp = keras.Input(shape=(None, 4))
    layers = [None] * (num_blocks + 1)
    layers[0] = keras.layers.Conv1D(model_dim, 1, activation='relu')(inp)
    for i in range(num_blocks):
        dilation = max(1, 2 ** (i // 2 - 1))
        layers[i + 1] = _metaformer(model_dim, kernel_size, dilation)(layers[i])

    outputs = []
    for j, od in enumerate(output_dims):
        head = keras.layers.Average()([
            keras.layers.Dense(od, activation='relu', name=f'output{j}_{i}')(layers[i])
            for i in range(num_blocks, 0, -shortcut_layer_freq)
        ])
        head = keras.layers.Cropping1D(cropping=output_crop // 2)(head)
        outputs.append(head)

    return keras.Model(inputs=inp, outputs=tuple(outputs))


def _save_savedmodel(model, path: str):
    """Save a tf_keras model in SavedModel format (directory, no extension)."""
    import tf_keras as keras
    keras.models.save_model(model, path, save_format='tf')


def test_convert_infers_architecture(tmp_path):
    """Converter correctly infers num_blocks, model_dim, output_dims from the Keras model."""
    from torch_promoterai.utils import convert_tf_weights

    model = _build_tf_keras_model(num_blocks=8, model_dim=16, output_dims=(8,))
    model_path = str(tmp_path / "keras_model")
    _save_savedmodel(model, model_path)

    out_pt = str(tmp_path / "model.pt")
    convert_tf_weights(model_path, out_pt)

    ckpt = torch.load(out_pt, map_location="cpu")
    assert ckpt["args"]["num_blocks"] == 8
    assert ckpt["args"]["model_dim"] == 16
    assert ckpt["args"]["output_dims"] == [8]


def test_convert_loads_cleanly(tmp_path):
    """Converted checkpoint loads into PromoterAI without missing keys."""
    from torch_promoterai.utils import convert_tf_weights, load_pretrained

    model = _build_tf_keras_model(num_blocks=8, model_dim=16, output_dims=(8,))
    model_path = str(tmp_path / "keras_model")
    _save_savedmodel(model, model_path)

    out_pt = str(tmp_path / "model.pt")
    convert_tf_weights(model_path, out_pt, input_length=512, output_length=512)

    pt_model, args = load_pretrained(out_pt)
    assert args["input_length"] == 512
    pt_model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 64, 4)
        x[:, :, 0] = 1.0
        out = pt_model(x)
    assert len(out) == 1


def test_convert_numerical_parity(tmp_path):
    """PyTorch and Keras models produce numerically matching outputs after weight conversion."""
    from torch_promoterai.utils import convert_tf_weights, load_pretrained

    input_len = 64
    keras_model = _build_tf_keras_model(num_blocks=8, model_dim=16, output_dims=(8,))
    model_path = str(tmp_path / "keras_model")
    _save_savedmodel(keras_model, model_path)

    out_pt = str(tmp_path / "model.pt")
    convert_tf_weights(model_path, out_pt)
    pt_model, _ = load_pretrained(out_pt)
    pt_model.eval()

    rng = np.random.default_rng(0)
    indices = rng.integers(0, 4, size=input_len)
    x_np = np.eye(4, dtype="float32")[indices][None]  # (1, L, 4)

    # Run inference through Keras model (call with training=False to use BN running stats)
    keras_out = keras_model(x_np, training=False)
    if isinstance(keras_out, (list, tuple)):
        keras_out = keras_out[0]
    keras_out = np.array(keras_out)  # (1, L, 8)

    # Run inference through PyTorch model
    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        pt_out = pt_model(x_pt)[0].numpy()  # (1, L, 8)

    np.testing.assert_allclose(
        pt_out, keras_out, atol=1e-4,
        err_msg="Keras and PyTorch outputs differ after conversion",
    )
