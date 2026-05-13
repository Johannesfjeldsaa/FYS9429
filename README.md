# FYS9429 - Advanced machine learning and data analysis for the physical sciences

In this course on ML I focus on emulators of earth system model (ESM) output.
I will be using the Multivariate Emulation of Time-Evolving and Overlapping Responses (METEOR) available at
(https://github.com/benmsanderson/METEOR?tab=readme-ov-file)[https://github.com/benmsanderson/METEOR?tab=readme-ov-file]
as a starting point and build a module for internal variability with daily resolution, i.e. temporal downscaling
from montly means to daily means.


## Machine: victor.uio.no

| Property | Value |
|----------|-------|
| Hostname | `victor.uio.no` |
| OS | RHEL 9.7 (Plow) |
| CPU | AMD EPYC 7763 64-Core (128 logical cores) |
| RAM | ~4 TB |
| GPU | 2× NVIDIA H100 NVL (96 GB each) |
| CUDA driver | 13.1 |
| Module system | Lmod 8.7.59 |

### Available HPC modules (used by `setup_fys9429.py --machine victor`)

| Module | Version |
|--------|---------|
| CUDA | `12.1.1` |
| cuDNN | `8.9.2.26-CUDA-12.1.1` |
| PyTorch | `2.1.2-foss-2023a-CUDA-12.1.1` |
| Miniforge3 | `24.11.3-0` |

> **Note:** `torchvision` is not available as a system module on victor and is installed via pip into the conda environment instead.

### Setup

```bash
module load Miniforge3/24.11.3-0
python setup_fys9429.py --machine victor --name fys9429 --register-kernel
conda activate fys9429
```
