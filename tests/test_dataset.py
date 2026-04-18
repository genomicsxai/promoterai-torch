import numpy as np
import pytest
from torch_promoterai.dataset import onehot_encode, _prepare_sample


def test_onehot_encode_bases():
    enc = onehot_encode('ACGT')
    assert enc.shape == (4, 4)
    assert enc[0].tolist() == [1, 0, 0, 0]  # A
    assert enc[1].tolist() == [0, 1, 0, 0]  # C
    assert enc[2].tolist() == [0, 0, 1, 0]  # G
    assert enc[3].tolist() == [0, 0, 0, 1]  # T


def test_onehot_encode_unknown():
    enc = onehot_encode('N')
    assert enc.shape == (1, 4)
    assert enc[0].tolist() == [0, 0, 0, 0]


def test_onehot_encode_lowercase():
    enc_upper = onehot_encode('ACGT')
    enc_lower = onehot_encode('acgt')
    np.testing.assert_array_equal(enc_upper, enc_lower)


def _make_xy(in_len=200, out_len=100, n_tracks=10):
    x = np.random.rand(in_len, 4).astype('float32')
    y = np.random.rand(out_len, n_tracks).astype('float32')
    return x, y


def test_prepare_sample_no_augment_shape():
    x, y = _make_xy(200, 100)
    target_in, target_out = 100, 50
    sw = (True,)
    x_crop, y_tuple, w_tuple = _prepare_sample(x, y, target_in, target_out, sw, augment=False)
    assert x_crop.shape == (target_in, 4)
    assert y_tuple[0].shape == (target_out, 10)


def test_prepare_sample_no_augment_no_strand_flip():
    """With augment=False, strand is always +1 (no reverse complement)."""
    x = onehot_encode('ACGTACGT' * 50)  # 400 bp
    y = np.random.rand(200, 5).astype('float32')
    sw = (True,)
    for _ in range(20):
        x_crop, y_tuple, _ = _prepare_sample(x, y, 200, 100, sw, augment=False)
        # First base should still be A (no RC)
        assert x_crop[0, 0] == 1.0, "Strand flip occurred with augment=False"


def test_prepare_sample_augment_strand_frequency():
    """With augment=True, ~25% of samples should be reverse complemented."""
    x = onehot_encode('AAAA' * 100)  # all-A, RC is all-T
    y = np.random.rand(200, 5).astype('float32')
    sw = (True,)
    np.random.seed(42)
    n_flipped = 0
    n_trials = 1000
    for _ in range(n_trials):
        x_crop, _, _ = _prepare_sample(x, y, 200, 100, sw, augment=True)
        # RC of all-A (index 0) is all-T (index 3)
        if x_crop[0, 0] == 0.0:
            n_flipped += 1
    frac = n_flipped / n_trials
    assert 0.18 < frac < 0.32, f"Expected ~25% strand flip, got {frac:.2%}"


def test_prepare_sample_species_weight():
    """sample_weight controls which y outputs are real vs dummy."""
    x, y = _make_xy(200, 100, 10)
    sw = (True, False)
    _, y_tuple, w_tuple = _prepare_sample(x, y, 100, 50, sw, augment=False)
    assert y_tuple[0].shape[-1] == 10   # real
    assert y_tuple[1].shape[-1] == 1    # dummy [[0.]]
    assert w_tuple[0] > 0               # weight > 0 for real
    assert w_tuple[1] == 0.0            # weight 0 for dummy


def test_prepare_sample_all_n_zero_weight():
    """All-N sequence encodes to all-zero → x.max()=0 → sample_weight=0."""
    x = onehot_encode('N' * 200)
    y = np.zeros((100, 5), dtype='float32')
    sw = (True,)
    _, _, w_tuple = _prepare_sample(x, y, 100, 50, sw, augment=False)
    assert w_tuple[0] == 0.0
