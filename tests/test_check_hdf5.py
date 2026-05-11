import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from promoterai_torch.check_hdf5 import check_hdf5_paths

ROOT = Path(__file__).resolve().parents[1]


def _write_hdf5(path, x_shape=(3, 8, 4), y_shape=(3, 4, 2)):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=np.ones(x_shape, dtype="float32"))
        handle.create_dataset("y", data=np.ones(y_shape, dtype="float32"))


def test_valid_hdf5_file_passes_and_reports_samples(tmp_path):
    path = tmp_path / "valid.h5"
    _write_hdf5(path)

    results, exit_code = check_hdf5_paths([tmp_path])

    assert exit_code == 0
    assert len(results) == 1
    assert results[0].ok
    assert results[0].samples == 3


def test_missing_required_dataset_fails(tmp_path):
    path = tmp_path / "missing_y.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=np.ones((2, 8, 4), dtype="float32"))

    results, exit_code = check_hdf5_paths([tmp_path])

    assert exit_code == 1
    assert len(results) == 1
    assert not results[0].ok
    assert "missing required dataset 'y'" in results[0].error


def test_mismatched_sample_counts_fail(tmp_path):
    path = tmp_path / "mismatch.h5"
    _write_hdf5(path, x_shape=(3, 8, 4), y_shape=(2, 4, 2))

    results, exit_code = check_hdf5_paths([path])

    assert exit_code == 1
    assert "sample counts differ" in results[0].error


def test_invalid_h5_file_fails_with_h5py_error(tmp_path):
    path = tmp_path / "truncated.h5"
    path.write_text("not an hdf5 file")

    results, exit_code = check_hdf5_paths([tmp_path])

    assert exit_code == 1
    assert not results[0].ok
    assert "OSError" in results[0].error


def test_empty_search_path_returns_no_files_status(tmp_path):
    results, exit_code = check_hdf5_paths([tmp_path])

    assert results == []
    assert exit_code == 2


def test_full_read_valid_file_passes(tmp_path):
    path = tmp_path / "valid.hdf"
    _write_hdf5(path)

    results, exit_code = check_hdf5_paths([tmp_path], full_read=True)

    assert exit_code == 0
    assert results[0].ok


def test_cli_smoke_returns_zero_for_valid_files(tmp_path):
    _write_hdf5(tmp_path / "valid.h5")

    result = _run_cli("check-hdf5", "--paths", str(tmp_path))

    assert result.returncode == 0
    assert "total files checked: 1" in result.stdout
    assert "total samples: 3" in result.stdout
    assert "bad file count: 0" in result.stdout


def test_cli_smoke_returns_nonzero_for_bad_files(tmp_path):
    _write_hdf5(tmp_path / "valid.h5")
    (tmp_path / "bad.h5").write_text("not an hdf5 file")

    result = _run_cli("check-hdf5", "--paths", str(tmp_path))

    assert result.returncode == 1
    assert "bad file count: 1" in result.stdout
    assert "BAD" in result.stdout
    assert "bad.h5" in result.stdout


def _run_cli(*args):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from promoterai_torch.cli import main; main()",
            *args,
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
