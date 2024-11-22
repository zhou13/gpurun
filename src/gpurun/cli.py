#!/usr/bin/env python3
import argparse
import contextlib
import os
import subprocess
import sys
import time

from filelock import FileLock, Timeout


def get_available_gpus():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return list(range(len(result.stdout.strip().split("\n"))))
    except (subprocess.SubprocessError, FileNotFoundError):
        print(
            "Warning: Could not detect GPUs using nvidia-smi. Defaulting to GPU 0",
            file=sys.stderr,
        )
        return [0]


class GPUManager:
    def __init__(
        self,
        gpu_ids: list[int],
        max_jobs_per_gpu: int,
        gpus_per_job: int = 1,
        session: str = "default-session",
    ):
        gpu_ids = gpu_ids * max_jobs_per_gpu
        self.gpu_ids = gpu_ids
        self.gpus_per_job = gpus_per_job
        self.session = session

        if len(gpu_ids) % gpus_per_job != 0:
            raise ValueError(
                f"Total number of GPU slots ({len(gpu_ids)}) must be a multiple of GPUs per job ({gpus_per_job})"
            )

        self.num_gpu_groups = len(gpu_ids) // gpus_per_job
        self.lock_files = [
            os.path.join(self.locks_dir, f"gpu_{group_id:02}.lock")
            for group_id in range(self.num_gpu_groups)
        ]

    @property
    def locks_dir(self) -> str:
        locks_dir = os.path.join(os.path.expanduser("~"), ".gpu_locks", self.session)
        os.makedirs(locks_dir, exist_ok=True)
        return locks_dir

    @contextlib.contextmanager
    def acquire_gpus(self):
        # Try to acquire GPU group in round-robin fashion
        while True:
            for group_id, lock_file in enumerate(self.lock_files):
                try:
                    with FileLock(lock_file).acquire(blocking=False):
                        start_idx = group_id * self.gpus_per_job
                        end_idx = start_idx + self.gpus_per_job
                        assigned_gpus = self.gpu_ids[start_idx:end_idx]
                        yield assigned_gpus
                        return
                except Timeout:
                    pass
            time.sleep(0.1)


def parse_gpu_arg(value):
    if value.lower() == "auto":
        return value.lower()
    try:
        gpu_ids = [int(x.strip()) for x in value.split(",")]
        return gpu_ids
    except ValueError:
        raise argparse.ArgumentTypeError(
            'GPU IDs must be "auto" or comma-separated integers (e.g., "0,1,2")'
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run a command with GPU management and scheduling")
    parser.add_argument(
        "--session",
        type=str,
        default="default-session",
        help='String representing the session name used in locks (default: "default-session")',
    )
    parser.add_argument(
        "--gpus",
        type=parse_gpu_arg,
        default="auto",
        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2") or "auto" for autodetection',
    )
    parser.add_argument(
        "-j",
        "--max-jobs-per-gpu",
        type=int,
        default=1,
        help="Maximum number of concurrent jobs allowed per GPU group (default: 1)",
    )
    parser.add_argument(
        "-g",
        "--gpus-per-job",
        type=int,
        default=1,
        help="Number of GPUs needed for each job (default: 1)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help='Command to run (e.g., "python run.py")',
    )
    args = parser.parse_args()

    if not args.command:
        parser.error("No command provided")

    return args


def main():
    args = parse_args()

    # Handle GPU autodetection
    if args.gpus == "auto":
        gpu_ids = get_available_gpus()
        print(f"Auto-detected GPUs: {gpu_ids}")
    else:
        gpu_ids = args.gpus

    # Initialize GPU manager
    gpu_manager = GPUManager(gpu_ids, args.max_jobs_per_gpu, args.gpus_per_job, args.session)

    try:
        with gpu_manager.acquire_gpus() as assigned_gpus:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, assigned_gpus))
            process = subprocess.run(args.command, env=env, check=True)
            sys.exit(process.returncode)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
