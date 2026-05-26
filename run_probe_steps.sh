#!/usr/bin/env bash
set -euo pipefail

PROBE_STEPS="${PROBE_STEPS:-1 5 10 15 20 25 30}"
OUTPUT_ROOT="${OUTPUT_ROOT:-layer_analysis_results}"
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

for PROBE_STEP in ${PROBE_STEPS}; do
  OUT_DIR="${OUTPUT_ROOT}/${PROBE_STEP}"
  echo "Running probe_step=${PROBE_STEP}; saving to ${OUT_DIR}"

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
    --probe-step "${PROBE_STEP}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH}" \
    --projection pca \
    --save-all-layer-projections
done

# Aggregate best layer per probe step into a single JSON file.
# Produces: ${OUTPUT_ROOT}/best_layers.json mapping probe_step -> best_layer
python - <<'PY'
import json, os

out_root = os.environ.get('OUTPUT_ROOT', 'layer_analysis_results')
probe_steps = os.environ.get('PROBE_STEPS', '1 5 10 15 20 25 30').split()
results = {}
for step in probe_steps:
  scores_path = os.path.join(out_root, step, 'layer_separability_scores.json')
  if not os.path.exists(scores_path):
    continue
  with open(scores_path, 'r', encoding='utf-8') as fh:
    scores = json.load(fh)
  if not scores:
    continue
  # scores keys may be strings; pick the key with the highest score
  best_layer = max(scores.items(), key=lambda kv: kv[1])[0]
  try:
    best_layer = int(best_layer)
  except Exception:
    pass
  results[str(step)] = best_layer

os.makedirs(out_root, exist_ok=True)
out_path = os.path.join(out_root, 'best_layers.json')
with open(out_path, 'w', encoding='utf-8') as fh:
  json.dump(results, fh, indent=2)
print('Wrote', out_path)
PY
