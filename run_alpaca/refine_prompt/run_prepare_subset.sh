#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-"${ROOT_DIR}/.venv/bin/python"}

"${PYTHON_BIN}" run_alpaca/refine_prompt/prepare_alpaca_subset.py \
  --sample-size 689 \
  --seed 42 \
  --output-json "${ROOT_DIR}/datasets/alpaca/alpaca_subset_689_seed42.json" \
  --output-csv "${ROOT_DIR}/run_alpaca/refine_prompt/alpaca_subset_689_seed42.csv"
