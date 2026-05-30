#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv312/bin/python}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-logs/exp_05_mahalanobis_euclidean}"
OUTPUT_ROOT="${OUTPUT_ROOT:-probe_outputs}"
HELDOUT_DIR="${HELDOUT_DIR:-logs/exp_04_hf_heldout_eval}"
PCA_DIM="${PCA_DIM:-32}"
LOG_FILE="${LOG_FILE:-${EXPERIMENT_DIR}/mahalanobis_heldout.log}"

mkdir -p "${EXPERIMENT_DIR}"

echo "Running PCA + Ledoit-Wolf Mahalanobis Euclidean baseline; log=${LOG_FILE}"

"${PYTHON}" evaluate_mahalanobis_euclidean.py \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --heldout-dir "${HELDOUT_DIR}" \
  --pca-dim "${PCA_DIM}" 2>&1 | tee "${LOG_FILE}"

echo "Mahalanobis logs, plots, and analysis written to ${EXPERIMENT_DIR}"
