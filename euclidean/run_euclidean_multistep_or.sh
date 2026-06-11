#!/usr/bin/env bash
set -euo pipefail

PROBE_STEPS="${PROBE_STEPS:-1 5 10 15 20 25 30}"
QUANTILES="${QUANTILES:-0.90 0.95 0.98 0.99}"
PYTHON="${PYTHON:-.venv312/bin/python}"
DEVICE="${DEVICE:-cpu}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-logs/exp_02_multistep_or_rule}"
OUTPUT_ROOT="${OUTPUT_ROOT:-probe_outputs}"
BEST_LAYERS_FILE="${BEST_LAYERS_FILE:-best_layers.json}"
LOG_FILE="${LOG_FILE:-${EXPERIMENT_DIR}/multistep_or.log}"

mkdir -p "${EXPERIMENT_DIR}"

echo "Running multi-step OR Euclidean baseline; log=${LOG_FILE}"

"${PYTHON}" evaluate_euclidean_multistep_or.py \
  --probe-steps ${PROBE_STEPS} \
  --quantiles ${QUANTILES} \
  --output-root "${OUTPUT_ROOT}" \
  --best-layers-file "${BEST_LAYERS_FILE}" \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --device "${DEVICE}" 2>&1 | tee "${LOG_FILE}"

echo "Multi-step OR logs, plots, and analysis written to ${EXPERIMENT_DIR}"
