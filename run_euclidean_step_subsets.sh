#!/usr/bin/env bash
set -euo pipefail

QUANTILES="${QUANTILES:-0.95 0.98}"
PYTHON="${PYTHON:-.venv312/bin/python}"
DEVICE="${DEVICE:-cpu}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-logs/exp_03_step_subset_or_ablation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-probe_outputs}"
BEST_LAYERS_FILE="${BEST_LAYERS_FILE:-best_layers.json}"
LOG_FILE="${LOG_FILE:-${EXPERIMENT_DIR}/step_subset_or.log}"

mkdir -p "${EXPERIMENT_DIR}"

echo "Running step-subset OR Euclidean ablation; log=${LOG_FILE}"

"${PYTHON}" evaluate_euclidean_step_subsets.py \
  --quantiles ${QUANTILES} \
  --output-root "${OUTPUT_ROOT}" \
  --best-layers-file "${BEST_LAYERS_FILE}" \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --device "${DEVICE}" 2>&1 | tee "${LOG_FILE}"

echo "Step-subset OR logs, plots, and analysis written to ${EXPERIMENT_DIR}"
