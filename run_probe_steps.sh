#!/usr/bin/env bash
set -euo pipefail

PROBE_STEPS="${PROBE_STEPS:-5 10 15}"
LAYER_ID="${LAYER_ID:-23}"
OUTPUT_ROOT="${OUTPUT_ROOT:-probe_outputs}"
PYTHON="${PYTHON:-.venv/bin/python}"

MODEL_NAME="${MODEL_NAME:-GSAI-ML/LLaDA-8B-Instruct}"
SAFE_LOCAL_PATH="${SAFE_LOCAL_PATH:-AlpacaDIJA/llada_instruct_DIJA_v1.json}"
SAFE_LOCAL_FIELD="${SAFE_LOCAL_FIELD:-Refined_behavior}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
STEPS="${STEPS:-64}"
GEN_LENGTH="${GEN_LENGTH:-64}"
BLOCK_LENGTH="${BLOCK_LENGTH:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"

OUT_DIR="${OUTPUT_ROOT}/layer_${LAYER_ID}"
echo "Extracting layer ${LAYER_ID} at steps ${PROBE_STEPS}; saving to ${OUT_DIR}"

"${PYTHON}" probe_analysis.py \
  --model-name "${MODEL_NAME}" \
  --safe-local-path "${SAFE_LOCAL_PATH}" \
  --safe-local-field "${SAFE_LOCAL_FIELD}" \
  --samples-per-class "${SAMPLES_PER_CLASS}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --output-dir "${OUT_DIR}" \
  --steps "${STEPS}" \
  --gen-length "${GEN_LENGTH}" \
  --block-length "${BLOCK_LENGTH}" \
  --probe-steps ${PROBE_STEPS} \
  --layer-id "${LAYER_ID}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH}"
