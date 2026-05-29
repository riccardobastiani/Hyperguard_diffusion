#!/usr/bin/env bash
set -euo pipefail

PROBE_STEPS="${PROBE_STEPS:-1 5 10 15 20 25 30}"
BEST_LAYERS_FILE="${BEST_LAYERS_FILE:-best_layers.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-probe_outputs}"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cpu}"
NU="${NU:-0.01}"
RADIUS_QUANTILE="${RADIUS_QUANTILE:-}"
L2_NORMALIZE_PROBES="${L2_NORMALIZE_PROBES:-0}"

if [[ ! -f "${BEST_LAYERS_FILE}" ]]; then
  echo "best layers file not found: ${BEST_LAYERS_FILE}" >&2
  exit 1
fi

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

  PROBE_DIR="${OUTPUT_ROOT}/step_${STEP}/layer_${LAYER_ID}"

  if [[ ! -f "${PROBE_DIR}/safe_probes.npz" || ! -f "${PROBE_DIR}/unsafe_probes.npz" ]]; then
    echo "probe cache missing in ${PROBE_DIR}" >&2
    echo "run run_probe_steps.sh first, or set OUTPUT_ROOT/PROBE_STEPS to an existing cache" >&2
    exit 1
  fi

  echo "Fitting Euclidean baseline: step ${STEP}, layer ${LAYER_ID}, cache ${PROBE_DIR}"

  CMD=("${PYTHON}" euclidean_baseline.py
    --probe-dir "${PROBE_DIR}"
    --output-dir "${PROBE_DIR}"
    --probe-steps "${STEP}"
    --nu "${NU}"
    --device "${DEVICE}")

  if [[ -n "${RADIUS_QUANTILE}" ]]; then
    CMD+=(--radius-quantile "${RADIUS_QUANTILE}")
  fi

  if [[ "${L2_NORMALIZE_PROBES}" == "1" ]]; then
    CMD+=(--l2-normalize-probes)
  fi

  "${CMD[@]}"
done
