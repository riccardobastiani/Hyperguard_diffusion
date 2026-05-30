#!/usr/bin/env bash
set -euo pipefail

PROBE_STEPS="${PROBE_STEPS:-1 5 10 15 20 25 30}"
QUANTILES="${QUANTILES:-0.90 0.95 0.98 0.99}"
PYTHON="${PYTHON:-.venv312/bin/python}"
DEVICE="${DEVICE:-cpu}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-logs/exp_01_threshold_sweep_quantiles}"
OUTPUT_ROOT="${OUTPUT_ROOT:-probe_outputs}"
BEST_LAYERS_FILE="${BEST_LAYERS_FILE:-best_layers.json}"

mkdir -p "${EXPERIMENT_DIR}"

read -r -a QUANTILE_LIST <<< "${QUANTILES}"

for Q in "${QUANTILE_LIST[@]}"; do
  SAFE_Q="${Q//./p}"
  LOG_FILE="${EXPERIMENT_DIR}/quantile_${SAFE_Q}.log"
  echo "Running Euclidean threshold sweep quantile=${Q}; log=${LOG_FILE}"

  PYTHON="${PYTHON}" \
  PROBE_STEPS="${PROBE_STEPS}" \
  DEVICE="${DEVICE}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  BEST_LAYERS_FILE="${BEST_LAYERS_FILE}" \
  LOG_FILE="${LOG_FILE}" \
  CALIBRATE_QUANTILE="${Q}" \
  ./run_euclidean_eval.sh
done

"${PYTHON}" summarize_threshold_sweep.py --experiment-dir "${EXPERIMENT_DIR}"

echo "Threshold sweep logs, plots, and analysis written to ${EXPERIMENT_DIR}"
