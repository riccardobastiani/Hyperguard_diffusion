#!/bin/bash
set -euo pipefail

cd /workspace/Hyperguard_diffusion

PYTHON_BIN=${PYTHON_BIN:-/workspace/Hyperguard_diffusion/.venv/bin/python}

"${PYTHON_BIN}" run_alpaca/refine_prompt/prepare_alpaca_subset.py \
  --sample-size 689 \
  --seed 42 \
  --output-json /workspace/Hyperguard_diffusion/datasets/alpaca/alpaca_subset_689_seed42.json \
  --output-csv /workspace/Hyperguard_diffusion/run_alpaca/refine_prompt/alpaca_subset_689_seed42.csv
