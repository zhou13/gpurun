# 🚀 GPU Runner (`gpurun`)

[![image](https://img.shields.io/pypi/v/gpurun.svg)](https://pypi.python.org/pypi/gpurun)
[![image](https://img.shields.io/pypi/l/gpurun.svg)](https://github.com/zhou13/gpurun/blob/main/LICENSE)
[![image](https://img.shields.io/pypi/pyversions/gpurun.svg)](https://pypi.python.org/pypi/gpurun)
[![Actions status](https://github.com/zhou13/gpurun/workflows/CI/badge.svg)](https://github.com/zhou13/gpurun/actions)

## Overview

`gpurun` is a sophisticated command-line tool designed to simplify GPU resource management and scheduling jobs on a single machine for machine learning, data science, and computational tasks. It provides intelligent GPU allocation, concurrency control, and seamless integration with CUDA applications using `CUDA_VISIBLE_DEVICES`. It is inspired by the `sem` tool from GNU parallel.

### Why Use GPU Runner?

Imagine you are on an 8-GPU machine, and you want to run 100 inference jobs. You want to run them in parallel. Each job needs 2 GPUs and each GPU can only run three jobs at a time to avoid GPU memory overflow. You think you need a cluster scheduler like `sbatch` but for a single machine on GPUs. If this is your problem, GPU Runner is the perfect solution for you!

**Think of GPU Runner as a local GPU scheduler that intelligently coordinates job execution, preventing memory overflow and maximizing computational efficiency.**

## 🌟 Key Features

- **Flexible GPU Allocation**: Automatically assign jobs to GPUs with `CUDA_VISIBLE_DEVICES`
- **Simple Interface**: Minimalistic command-line interface
- **Concurrent Job Management**: Limit and control GPU usage across multiple jobs
- **Automatic GPU Detection**: Automatically identifies available GPUs
- **GNU Parallel Compatibility**: Easy to be used with GNU parallel
- **Session-based Locking**: Prevent GPU contention between different computational tasks

## Installation

```bash
pip install git+https://github.com/zhou13/gpurun
```

## Usage Examples

### Basic Usage

```bash
# 1. Run infer.py (needs 1 GPU) on 100 images with all the GPUs in parallel, 1 job per GPU.
for i in $(seq 1 100); do gpurun python infer.py $i.jpg & done

# 2. Same as 1, but put 2 jobs per GPU at the same time.
for i in $(seq 1 100); do gpurun -j2 python infer.py $i.jpg & done

# 3. Same as 2, but use gnu-parallel to simplify the command.
parallel -j0 gpurun -j2 python infer.py {} ::: $(seq 1 100)

# 4. Same as 1, but infer.py now will see 2 GPUs.
parallel -j0 gpurun -g2 python infer.py {} ::: $(seq 1 100)

# 5. You can customize the GPUs to be used with --gpus.
parallel -j0 gpurun --gpus 0,1 python infer.py {} ::: $(seq 1 100)

# 6. You can customize the name of lockfile with --session.
parallel -j0 gpurun --session ml-seesion python infer.py {} ::: $(seq 1 100)
```

## Configuration Options

```
$ gpurun --help
usage: gpurun [-h] [--session SESSION] [--gpus GPUS] [-j MAX_JOBS_PER_GPU] [-g GPUS_PER_JOB] ...

Run a command with GPU management and scheduling

positional arguments:
  command               Command to run (e.g., "python run.py")

options:
  -h, --help            show this help message and exit
  --session SESSION     String representing the session name used in locks (default: "default-session")
  --gpus GPUS           Comma-separated list of GPU IDs to use (e.g., "0,1,2") or "auto" for autodetection
  -j MAX_JOBS_PER_GPU, --max-jobs-per-gpu MAX_JOBS_PER_GPU
                        Maximum number of concurrent jobs allowed per GPU group (default: 1)
  -g GPUS_PER_JOB, --gpus-per-job GPUS_PER_JOB
                        Number of GPUs needed for each job (default: 1)
```

## Contributing

Contributions welcome! Feel free to open issues and submit pull requests.

## License

This project is licensed under the MIT License.
