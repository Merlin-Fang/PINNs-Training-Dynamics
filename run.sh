#!/usr/bin/env bash
set -euo pipefail

# ----- choose GPUs -----
export CUDA_VISIBLE_DEVICES="0, 1, 2, 3, 4"

# ----- determinism / reproducibility -----
export TF_CUDNN_DETERMINISTIC="1"

# ----- run -----
python -m src.main --config ./src/configs/default.py
