#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
GPO_V_ROOT="${1:-${GPO_V_ROOT:-}}"
UNSAFE_FILE="${2:-${UNSAFE_FILE:-}}"
UNSAFE_FIELD="${UNSAFE_FIELD:-}"
IMAGE_PATH="${IMAGE_PATH:-gpo_v/assets/example_input_image.jpg}"
DATASET_PATH="${DATASET_PATH:-data/multimodal_gpo/smoke.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/multimodal_gpo/smoke_skip_attack}"
GENERATION_STEPS="${GENERATION_STEPS:-32}"
GEN_LENGTH="${GEN_LENGTH:-32}"
BLOCK_LENGTH="${BLOCK_LENGTH:-8}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
QUANTIZATION="${QUANTIZATION:-4bit}"

cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env.multimodal" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env.multimodal"
  GPO_V_ROOT="${1:-${GPO_V_ROOT:-}}"
fi

if [[ "$VENV_DIR" == "$ROOT_DIR/.venv" && ! -f "$VENV_DIR/bin/activate" && -f "$ROOT_DIR/venv/bin/activate" ]]; then
  VENV_DIR="$ROOT_DIR/venv"
fi

if [[ -z "$GPO_V_ROOT" || ! -f "$GPO_V_ROOT/llava/model/builder.py" ]]; then
  echo "Usage: bash scripts/run_multimodal_smoke.sh /path/to/GPO-V/LLaDA-V/train [unsafe_prompts.csv]" >&2
  echo "GPO_V_ROOT must point to the upstream LLaDA-V train folder containing llava/model/builder.py." >&2
  exit 1
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Missing virtualenv at $VENV_DIR. Run scripts/setup_ssh_env.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$(dirname "$DATASET_PATH")" "$OUTPUT_DIR"

quant_flags=()
if [[ "$QUANTIZATION" == "8bit" ]]; then
  quant_flags+=(--load-8bit)
elif [[ "$QUANTIZATION" == "4bit" ]]; then
  quant_flags+=(--load-4bit)
fi

prepare_cmd=(
  python prepare_multimodal_gpo_dataset.py
  --max-safe 1
  --max-unsafe 1
  --image-path "$IMAGE_PATH"
  --output "$DATASET_PATH"
)

if [[ -n "$UNSAFE_FILE" ]]; then
  if [[ ! -f "$UNSAFE_FILE" ]]; then
    echo "Unsafe prompt file not found: $UNSAFE_FILE" >&2
    exit 1
  fi
  prepare_cmd+=(--unsafe-file "$UNSAFE_FILE" --unsafe-source-name smoke_unsafe)
  if [[ -n "$UNSAFE_FIELD" ]]; then
    prepare_cmd+=(--unsafe-field "$UNSAFE_FIELD")
  fi
fi

"${prepare_cmd[@]}"

python run_gpo_v_lladav_probe.py \
  --gpo-v-root "$GPO_V_ROOT" \
  --input-jsonl "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --device-map "$DEVICE_MAP" \
  --probe-steps 5 \
  --layer-ids 23 \
  --max-samples 2 \
  --generation-steps "$GENERATION_STEPS" \
  --gen-length "$GEN_LENGTH" \
  --block-length "$BLOCK_LENGTH" \
  "${quant_flags[@]}" \
  --skip-attack

OUTPUT_DIR="$OUTPUT_DIR" python - <<'PY'
import os
from pathlib import Path
import json
import numpy as np

output_dir = Path(os.environ["OUTPUT_DIR"])
npz_path = output_dir / "probes.npz"
metadata_path = output_dir / "metadata.json"

if not npz_path.exists():
    raise SystemExit(f"Missing probe file: {npz_path}")
if not metadata_path.exists():
    raise SystemExit(f"Missing metadata file: {metadata_path}")

arrays = np.load(npz_path)
required_keys = {"labels", "step_5_layer_23"}
missing = required_keys.difference(arrays.files)
if missing:
    raise SystemExit(f"Missing probe keys: {sorted(missing)}")

features = arrays["step_5_layer_23"]
if features.ndim != 2:
    raise SystemExit(f"Expected 2D features, got shape {features.shape}")
if not np.isfinite(features).all():
    raise SystemExit("Features contain NaN or Inf values")

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
print(f"Smoke probes ok: samples={len(metadata)} feature_shape={features.shape}")
PY

if [[ "${RUN_ATTACK_SMOKE:-0}" == "1" ]]; then
  ATTACK_OUTPUT_DIR="${ATTACK_OUTPUT_DIR:-outputs/multimodal_gpo/smoke_with_attack}"
  mkdir -p "$ATTACK_OUTPUT_DIR"
  python run_gpo_v_lladav_probe.py \
    --gpo-v-root "$GPO_V_ROOT" \
    --input-jsonl "$DATASET_PATH" \
    --output-dir "$ATTACK_OUTPUT_DIR" \
    --device-map "$DEVICE_MAP" \
    --probe-steps 5 \
    --layer-ids 23 \
    --max-samples 2 \
    --generation-steps "$GENERATION_STEPS" \
    --gen-length "$GEN_LENGTH" \
    --block-length "$BLOCK_LENGTH" \
    "${quant_flags[@]}" \
    --attack-steps "${ATTACK_STEPS:-5}"
fi

echo "Smoke run finished."
echo "For the real attacked run, omit --skip-attack in run_gpo_v_lladav_probe.py."
