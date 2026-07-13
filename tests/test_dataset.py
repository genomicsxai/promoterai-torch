import numpy as np
import pytest  # noqa
import pandas as pd
import pyfaidx

from promoterai_torch.dataset import (
    SequenceDataset,
    VariantDataset,
    _prepare_sample,
    build_weighted_dataloader,
    onehot_encode,
)
import promoterai_torch.dataset as dataset_module


def _require_h5py():
    return pytest.importorskip("h5py")


def test_onehot_encode_bases():
    enc = onehot_encode("ACGT")
    assert enc.shape == (4, 4)
    assert enc[0].tolist() == [1, 0, 0, 0]  # A
    assert enc[1].tolist() == [0, 1, 0, 0]  # C
    assert enc[2].tolist() == [0, 0, 1, 0]  # G
    assert enc[3].tolist() == [0, 0, 0, 1]  # T


def test_onehot_encode_unknown():
    enc = onehot_encode("N")
    assert enc.shape == (1, 4)
    assert enc[0].tolist() == [0, 0, 0, 0]


def test_onehot_encode_lowercase():
    enc_upper = onehot_encode("ACGT")
    enc_lower = onehot_encode("acgt")
    np.testing.assert_array_equal(enc_upper, enc_lower)


def _make_xy(in_len=200, out_len=100, n_tracks=10):
    x = np.random.rand(in_len, 4).astype("float32")
    y = np.random.rand(out_len, n_tracks).astype("float32")
    return x, y


def test_prepare_sample_no_augment_shape():
    x, y = _make_xy(200, 100)
    target_in, target_out = 100, 50
    sw = (True,)
    x_crop, y_tuple, w_tuple = _prepare_sample(
        x, y, target_in, target_out, sw, augment=False
    )
    assert x_crop.shape == (target_in, 4)
    assert y_tuple[0].shape == (target_out, 10)


def test_prepare_sample_no_augment_no_strand_flip():
    """With augment=False, strand is always +1 (no reverse complement)."""
    x = onehot_encode("ACGTACGT" * 50)  # 400 bp
    y = np.random.rand(200, 5).astype("float32")
    sw = (True,)
    for _ in range(20):
        x_crop, y_tuple, _ = _prepare_sample(x, y, 200, 100, sw, augment=False)
        # First base should still be A (no RC)
        assert x_crop[0, 0] == 1.0, "Strand flip occurred with augment=False"


def test_prepare_sample_augment_strand_frequency():
    """With augment=True, ~25% of samples should be reverse complemented."""
    x = onehot_encode("AAAA" * 100)  # all-A, RC is all-T
    y = np.random.rand(200, 5).astype("float32")
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
    assert y_tuple[0].shape[-1] == 10  # real
    assert y_tuple[1].shape[-1] == 1  # dummy [[0.]]
    assert w_tuple[0] > 0  # weight > 0 for real
    assert w_tuple[1] == 0.0  # weight 0 for dummy


def test_prepare_sample_all_n_zero_weight():
    """All-N sequence encodes to all-zero → x.max()=0 → sample_weight=0."""
    x = onehot_encode("N" * 200)
    y = np.zeros((100, 5), dtype="float32")
    sw = (True,)
    _, _, w_tuple = _prepare_sample(x, y, 100, 50, sw, augment=False)
    assert w_tuple[0] == 0.0


def test_sequence_dataset_reverse_augmentation_returns_positive_stride_tensors(
    tmp_path, monkeypatch
):
    h5py = _require_h5py()
    h5_path = tmp_path / "train.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("x", data=onehot_encode("ACGT" * 50)[None, :, :])
        handle.create_dataset(
            "y", data=np.arange(200, dtype="float32").reshape(1, 100, 2)
        )
    monkeypatch.setattr(dataset_module, "_truncated_normal", lambda stddev: 0.0)
    monkeypatch.setattr(dataset_module.np.random, "uniform", lambda: 0.0)

    ds = SequenceDataset([str(h5_path)], 100, 50, (True,), augment=True)
    x_t, y_t, _ = ds[0]

    assert x_t.shape == (100, 4)
    assert y_t[0].shape == (50, 2)
    assert all(stride > 0 for stride in x_t.stride())
    assert all(stride > 0 for stride in y_t[0].stride())


def test_sequence_dataset_caches_hdf5_handles(tmp_path):
    h5py = _require_h5py()
    h5_path = tmp_path / "train.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("x", data=np.zeros((2, 200, 4), dtype="float32"))
        handle.create_dataset("y", data=np.zeros((2, 100, 2), dtype="float32"))

    ds = SequenceDataset([str(h5_path)], 100, 50, (True,))
    ds[0]
    first_handle = ds._handles[0]
    ds[1]

    assert ds._handles[0] is first_handle
    ds.close()
    assert ds._handles == {}


def test_weighted_dataloader_persistent_workers_and_prefetch_options():
    loader = build_weighted_dataloader(
        [_IndexDataset(0, 5)],
        batch_size=1,
        num_workers=1,
        prefetch_factor=3,
    )

    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 3


def test_weighted_dataloader_omits_worker_prefetch_options_without_workers():
    loader = build_weighted_dataloader(
        [_IndexDataset(0, 5)],
        batch_size=1,
        num_workers=0,
        prefetch_factor=3,
    )

    assert loader.num_workers == 0
    assert loader.prefetch_factor is None


def test_variant_dataset_boundary_zeros_matches_tf_generator(tmp_path):
    fasta_path = tmp_path / "mini.fa"
    fasta_path.write_text(">chr1\nACGTACGT\n")
    fasta = pyfaidx.Fasta(str(fasta_path))
    df = pd.DataFrame(
        [{"chrom": "chr1", "pos": 2, "ref": "C", "alt": "A", "z": 1.0}]
    )

    ds = VariantDataset(df, fasta, input_length=8, output_col="z", boundary="zeros")
    (x_ref, x_alt), y = ds[0]

    assert y == 0.0
    assert x_ref.sum() == 0.0
    assert x_alt.sum() == 0.0


def test_variant_dataset_default_padding_still_supports_scoring_near_boundary(tmp_path):
    fasta_path = tmp_path / "mini.fa"
    fasta_path.write_text(">chr1\nACGTACGT\n")
    fasta = pyfaidx.Fasta(str(fasta_path))
    df = pd.DataFrame(
        [{"chrom": "chr1", "pos": 2, "ref": "C", "alt": "A", "z": 1.0}]
    )

    ds = VariantDataset(df, fasta, input_length=8, output_col="z")
    (x_ref, x_alt), _ = ds[0]

    assert x_ref.sum() > 0.0
    assert x_alt.sum() > 0.0


def test_variant_dataset_rejects_non_acgt_alt(tmp_path):
    fasta_path = tmp_path / "mini.fa"
    fasta_path.write_text(">chr1\nACGTACGT\n")
    fasta = pyfaidx.Fasta(str(fasta_path))
    df = pd.DataFrame(
        [{"chrom": "chr1", "pos": 5, "ref": "A", "alt": "N", "z": 1.0}]
    )

    ds = VariantDataset(df, fasta, input_length=4, output_col="z")
    (x_ref, x_alt), _ = ds[0]

    assert x_ref.sum() == 0.0
    assert x_alt.sum() == 0.0


class _IndexDataset:
    def __init__(self, start, length):
        self.start = start
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.start + idx


def test_weighted_dataloader_samples_datasets_proportional_to_size():
    loader = build_weighted_dataloader(
        [_IndexDataset(0, 2), _IndexDataset(2, 4)],
        batch_size=1,
        num_workers=0,
        num_samples=6000,
    )

    values = [int(batch.item()) for batch in loader]
    first_dataset = sum(value < 2 for value in values) / len(values)
    second_dataset = sum(value >= 2 for value in values) / len(values)

    assert first_dataset == pytest.approx(2 / 6, abs=0.04)
    assert second_dataset == pytest.approx(4 / 6, abs=0.04)


def test_validation_dataloader_is_sequential_without_replacement():
    loader = build_weighted_dataloader(
        [_IndexDataset(0, 5)],
        batch_size=2,
        num_workers=0,
        shuffle=False,
    )

    values = []
    for batch in loader:
        values.extend(batch.tolist())

    assert values == [0, 1, 2, 3, 4]
    assert loader.sampler is None or not hasattr(loader.sampler, "replacement")


def test_rank_local_weighted_sampler_uses_equal_sample_counts():
    loader0 = build_weighted_dataloader(
        [_IndexDataset(0, 2), _IndexDataset(2, 4)],
        batch_size=2,
        num_workers=0,
        rank=0,
        world_size=2,
        num_samples=7,
    )
    loader1 = build_weighted_dataloader(
        [_IndexDataset(0, 2), _IndexDataset(2, 4)],
        batch_size=2,
        num_workers=0,
        rank=1,
        world_size=2,
        num_samples=7,
    )

    assert loader0.sampler.num_samples == 7
    assert loader1.sampler.num_samples == 7
