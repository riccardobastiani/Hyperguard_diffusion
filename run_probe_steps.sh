#!/usr/bin/env bash
set -euo pipefail

PROBE_STEPS="${PROBE_STEPS:-1 5 10 20 30 40 50 64}"
OUTPUT_ROOT="${OUTPUT_ROOT:-probe_outputs}"
PYTHON="${PYTHON:-.venv/bin/python}"

MODEL_NAME="${MODEL_NAME:-GSAI-ML/LLaDA-8B-Instruct}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
STEPS="${STEPS:-64}"
GEN_LENGTH="${GEN_LENGTH:-64}"
BLOCK_LENGTH="${BLOCK_LENGTH:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"

for PROBE_STEP in ${PROBE_STEPS}; do
  OUT_DIR="${OUTPUT_ROOT}/${PROBE_STEP}"
  echo "Running probe_step=${PROBE_STEP}; saving to ${OUT_DIR}"

  "${PYTHON}" probe_analysis.py \
    --model-name "${MODEL_NAME}" \
    --samples-per-class "${SAMPLES_PER_CLASS}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --output-dir "${OUT_DIR}" \
    --steps "${STEPS}" \
    --gen-length "${GEN_LENGTH}" \
    --block-length "${BLOCK_LENGTH}" \
    --probe-step "${PROBE_STEP}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH}" \
    --projection pca \
    --save-all-layer-projections
done
