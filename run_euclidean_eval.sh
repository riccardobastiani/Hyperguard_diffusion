#!/usr/bin/env bash
set -euo pipefail

PROBE_STEPS="${PROBE_STEPS:-1 5 10 15 20 25 30}"
BEST_LAYERS_FILE="${BEST_LAYERS_FILE:-best_layers.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-probe_outputs}"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cpu}"
LOG_DIR="${LOG_DIR:-logs/manual_runs}"
LOG_FILE="${LOG_FILE:-}"
HF_TEST="${HF_TEST:-0}"
HF_SPLIT="${HF_SPLIT:-test}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-50}"
SEED="${SEED:-42}"
MODEL_NAME="${MODEL_NAME:-GSAI-ML/LLaDA-8B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-1}"
STEPS="${STEPS:-64}"
GEN_LENGTH="${GEN_LENGTH:-64}"
BLOCK_LENGTH="${BLOCK_LENGTH:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
CALIBRATE_QUANTILE="${CALIBRATE_QUANTILE:-}"
OVERWRITE_CHECKPOINTS="${OVERWRITE_CHECKPOINTS:-0}"

if [[ ! -f "${BEST_LAYERS_FILE}" ]]; then
  echo "best layers file not found: ${BEST_LAYERS_FILE}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
if [[ -z "${LOG_FILE}" ]]; then
  LOG_FILE="${LOG_DIR}/euclidean_eval_$(date +%Y%m%d_%H%M%S).log"
fi

echo "Writing Euclidean evaluation log to ${LOG_FILE}"

read -r -a STEP_LIST <<< "${PROBE_STEPS}"

for STEP in "${STEP_LIST[@]}"; do
  LAYER_ID="$("${PYTHON}" - "${BEST_LAYERS_FILE}" "${STEP}" <<'PY'
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

  CKPT_DIR="${OUTPUT_ROOT}/step_${STEP}/layer_${LAYER_ID}"

  if [[ ! -f "${CKPT_DIR}/euclidean_step_${STEP}.pt" ]]; then
    echo "Euclidean checkpoint missing: ${CKPT_DIR}/euclidean_step_${STEP}.pt" >&2
    echo "run run_euclidean_baseline.sh first" >&2
    exit 1
  fi

  echo "Evaluating Euclidean baseline: step ${STEP}, layer ${LAYER_ID}, dir ${CKPT_DIR}" | tee -a "${LOG_FILE}"

  CMD=("${PYTHON}" test_guard.py
    --detector-type euclidean
    --ckpt-dir "${CKPT_DIR}"
    --probe-steps "${STEP}"
    --eval-all-safe
    --eval-all-unsafe
    --metrics
    --device "${DEVICE}")

  if [[ "${HF_TEST}" == "1" ]]; then
    CMD+=(--hf-test
      --hf-split "${HF_SPLIT}"
      --samples-per-class "${SAMPLES_PER_CLASS}"
      --seed "${SEED}"
      --model-name "${MODEL_NAME}"
      --layer-id "${LAYER_ID}"
      --batch-size "${BATCH_SIZE}"
      --steps "${STEPS}"
      --gen-length "${GEN_LENGTH}"
      --block-length "${BLOCK_LENGTH}"
      --max-prompt-length "${MAX_PROMPT_LENGTH}")
  fi

  if [[ -n "${CALIBRATE_QUANTILE}" ]]; then
    CMD+=(--calibrate-quantile "${CALIBRATE_QUANTILE}")
  fi

  if [[ "${OVERWRITE_CHECKPOINTS}" == "1" ]]; then
    CMD+=(--overwrite-checkpoints)
  fi

  "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
done

echo "Euclidean evaluation log saved to ${LOG_FILE}"
