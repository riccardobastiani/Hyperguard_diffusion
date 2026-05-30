#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv312/bin/python}"
DEVICE="${DEVICE:-cuda}"
GPU_ID="${GPU_ID:-}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-logs/exp_04_hf_heldout_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-probe_outputs}"
HF_SPLIT="${HF_SPLIT:-test}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-50}"
SEED="${SEED:-42}"
MODEL_NAME="${MODEL_NAME:-GSAI-ML/LLaDA-8B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-1}"
STEPS="${STEPS:-64}"
GEN_LENGTH="${GEN_LENGTH:-64}"
BLOCK_LENGTH="${BLOCK_LENGTH:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
REUSE_CACHE="${REUSE_CACHE:-0}"
LOG_FILE="${LOG_FILE:-${EXPERIMENT_DIR}/hf_heldout.log}"

mkdir -p "${EXPERIMENT_DIR}"

CMD=("${PYTHON}" evaluate_euclidean_hf_heldout.py
  --experiment-dir "${EXPERIMENT_DIR}"
  --output-root "${OUTPUT_ROOT}"
  --hf-split "${HF_SPLIT}"
  --samples-per-class "${SAMPLES_PER_CLASS}"
  --seed "${SEED}"
  --model-name "${MODEL_NAME}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --steps "${STEPS}"
  --gen-length "${GEN_LENGTH}"
  --block-length "${BLOCK_LENGTH}"
  --max-prompt-length "${MAX_PROMPT_LENGTH}")

if [[ -n "${GPU_ID}" ]]; then
  CMD+=(--gpu-id "${GPU_ID}")
fi

if [[ "${REUSE_CACHE}" == "1" ]]; then
  CMD+=(--reuse-cache)
fi

echo "Running held-out HF Euclidean evaluation; log=${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

echo "Held-out HF logs, plots, and analysis written to ${EXPERIMENT_DIR}"
