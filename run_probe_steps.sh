#!/usr/bin/env bash
set -euo pipefail

PROBE_STEPS="${PROBE_STEPS:-1 5 10 15 20 25 30}"
BEST_LAYERS_FILE="${BEST_LAYERS_FILE:-best_layers.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-probe_outputs}"
PYTHON="${PYTHON:-.venv/bin/python}"

MODEL_NAME="${MODEL_NAME:-GSAI-ML/LLaDA-8B-Instruct}"
SAFE_LOCAL_PATH="${SAFE_LOCAL_PATH:-}"
SAFE_LOCAL_FIELD="${SAFE_LOCAL_FIELD:-Refined_behavior}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cuda}"
GPU_ID="${GPU_ID:-1}"
SEED="${SEED:-42}"
STEPS="${STEPS:-64}"
GEN_LENGTH="${GEN_LENGTH:-64}"
BLOCK_LENGTH="${BLOCK_LENGTH:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
L2_NORMALIZE_PROBES="${L2_NORMALIZE_PROBES:-1}"

# Projection + SVDD hyperparameters
PROJ_DIM="${PROJ_DIM:-128}"
CURVATURE="${CURVATURE:-1.0}"
NU="${NU:-0.01}"
SVDD_EPOCHS="${SVDD_EPOCHS:-100}"
SVDD_LR="${SVDD_LR:-1e-4}"
SVDD_BATCH_SIZE="${SVDD_BATCH_SIZE:-32}"

if [[ ! -f "${BEST_LAYERS_FILE}" ]]; then
  echo "best layers file not found: ${BEST_LAYERS_FILE}" >&2
  exit 1
fi

read -r -a STEP_LIST <<< "${PROBE_STEPS}"

for STEP in "${STEP_LIST[@]}"; do
  LAYER_ID="$(${PYTHON} - "${BEST_LAYERS_FILE}" "${STEP}" <<'PY'
import json
import sys

path = sys.argv[1]
step = sys.argv[2]

with open(path, "r", encoding="utf-8") as handle:
    mapping = json.load(handle)

if step not in mapping:
    raise SystemExit(f"No best-layer entry found for step {step} in {path}")

print(int(mapping[step]))
PY
  )"

  OUT_DIR="${OUTPUT_ROOT}/step_${STEP}/layer_${LAYER_ID}"
  echo "Running full pipeline: step ${STEP}, layer ${LAYER_ID}, saving to ${OUT_DIR}"

  CMD=("${PYTHON}" probe_analysis.py \
    --model-name "${MODEL_NAME}" \
    --safe-local-field "${SAFE_LOCAL_FIELD}" \
    --samples-per-class "${SAMPLES_PER_CLASS}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --gpu-id "${GPU_ID}" \
    --seed "${SEED}" \
    --output-dir "${OUT_DIR}" \
    --steps "${STEPS}" \
    --gen-length "${GEN_LENGTH}" \
    --block-length "${BLOCK_LENGTH}" \
    --probe-steps "${STEP}" \
    --layer-id "${LAYER_ID}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH}" \
    --proj-dim "${PROJ_DIM}" \
    --curvature "${CURVATURE}" \
    --nu "${NU}" \
    --svdd-epochs "${SVDD_EPOCHS}" \
    --svdd-lr "${SVDD_LR}" \
    --svdd-batch-size "${SVDD_BATCH_SIZE}")

  if [[ -n "${SAFE_LOCAL_PATH}" ]]; then
    CMD+=(--safe-local-path "${SAFE_LOCAL_PATH}")
  fi

  if [[ "${L2_NORMALIZE_PROBES}" == "1" ]]; then
    CMD+=(--l2-normalize-probes)
  fi

  "${CMD[@]}"
done