import argparse
import os
import subprocess
import sys
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

import gpurun.cli as cli


def test_gpu_detection_success():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="GPU1\nGPU2\nGPU3", text=True)
        gpus = cli.get_available_gpus()
        assert gpus == [0, 1, 2]


def test_gpu_detection_failure():
    with patch("subprocess.run", side_effect=subprocess.SubprocessError):
        gpus = cli.get_available_gpus()
        assert gpus == [0]


def test_gpu_manager_init():
    manager = cli.GPUManager([0, 1, 2, 3, 4], max_jobs_per_gpu=2, gpus_per_job=2)
    assert manager.num_gpu_groups == 5
    with pytest.raises(ValueError):
        cli.GPUManager([0, 1, 2], max_jobs_per_gpu=2, gpus_per_job=5)


def test_lock_file_creation(tmp_path):
    with patch.object(
        cli.GPUManager,
        "locks_dir",
        new_callable=PropertyMock,
        return_value=str(tmp_path),
    ):
        _ = cli.GPUManager([0, 1, 2, 3], max_jobs_per_gpu=2, gpus_per_job=2, session="test")
        assert tmp_path.exists()
        assert tmp_path.is_dir()


def test_gpu_acquisition(tmp_path):
    with patch.object(
        cli.GPUManager,
        "locks_dir",
        new_callable=PropertyMock,
        return_value=str(tmp_path),
    ):
        manager = cli.GPUManager([0, 1, 2, 3], max_jobs_per_gpu=2, gpus_per_job=2)

        with manager.acquire_gpus() as gpus0:
            assert gpus0 == [0, 1]
            with manager.acquire_gpus() as gpus1:
                assert gpus1 == [2, 3]
            with manager.acquire_gpus() as gpus1:
                assert gpus1 == [2, 3]
                with manager.acquire_gpus() as gpus2:
                    assert gpus2 == [0, 1]
                    with manager.acquire_gpus() as gpus3:
                        assert gpus3 == [2, 3]


def test_gpu_arg_parsing():
    # Test auto detection
    assert cli.parse_gpu_arg("auto") == "auto"

    # Test valid GPU list
    assert cli.parse_gpu_arg("0,1,2") == [0, 1, 2]

    # Test invalid GPU list
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_gpu_arg("a,b,c")


def test_cli_integration(tmp_path):
    # Create a simple test script
    test_script = tmp_path / "test_script.py"
    test_script.write_text("""
import os
import sys

# Verify CUDA_VISIBLE_DEVICES is set
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    sys.exit(1)

# Verify the environment variable contains valid GPU IDs
gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
if not all(gpu.isdigit() for gpu in gpus):
    sys.exit(2)

sys.exit(0)
""")

    # Run the CLI with the test script
    result = subprocess.run(
        [
            sys.executable,
            cli.__file__,
            "--gpus",
            "0,1",
            sys.executable,
            str(test_script),
        ],
        capture_output=True,
    )
    print(os.path.dirname(os.path.abspath(__file__)) + "/cli.py")

    assert result.returncode == 0
