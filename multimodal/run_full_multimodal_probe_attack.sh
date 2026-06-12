#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
ARG1="${1:-}"
ARG2="${2:-}"
GPO_V_ROOT="${GPO_V_ROOT:-}"
UNSAFE_FILE="${UNSAFE_FILE:-}"

SAFE_FILE="${SAFE_FILE:-}"
SAFE_FIELD="${SAFE_FIELD:-}"
UNSAFE_FIELD="${UNSAFE_FIELD:-}"
IMAGE_PATH="${IMAGE_PATH:-gpo_v/assets/example_input_image.jpg}"
DATASET_PATH="${DATASET_PATH:-data/multimodal_gpo/llada_v_gpo_prompts_400.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-probes_400_attack_all_layers_steps_1_5_10_15}"
LOG_DIR="${LOG_DIR:-results/multimodal_gpo_full_attack}"

MAX_SAFE="${MAX_SAFE:-200}"
MAX_UNSAFE="${MAX_UNSAFE:-200}"
PROBE_STEPS="${PROBE_STEPS:-1 5 10 15}"
ATTACK_STEPS="${ATTACK_STEPS:-5}"
GENERATION_STEPS="${GENERATION_STEPS:-128}"
GEN_LENGTH="${GEN_LENGTH:-128}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
DTYPE="${DTYPE:-float16}"
QUANTIZATION="${QUANTIZATION:-}"
SAVE_EVERY="${SAVE_EVERY:-25}"
SEED="${SEED:-42}"
RESUME="${RESUME:-1}"

cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env.multimodal" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env.multimodal"
fi

if [[ -n "$ARG2" ]]; then
  GPO_V_ROOT="$ARG1"
  UNSAFE_FILE="$ARG2"
elif [[ -n "$ARG1" ]]; then
  if [[ -n "$GPO_V_ROOT" && -f "$GPO_V_ROOT/llava/model/builder.py" ]]; then
    UNSAFE_FILE="$ARG1"
  else
    GPO_V_ROOT="$ARG1"
  fi
fi

if [[ -z "$GPO_V_ROOT" || ! -f "$GPO_V_ROOT/llava/model/builder.py" ]]; then
  echo "Usage: bash run_full_multimodal_probe_attack.sh [/path/to/GPO-V/LLaDA-V] /path/to/advbench.csv" >&2
  echo "Run scripts/download_gpo_v_upstream.sh first to create external/GPO-V-0250/LLaDA-V." >&2
  echo "GPO_V_ROOT must contain llava/model/builder.py." >&2
  exit 1
fi

if [[ -z "$UNSAFE_FILE" || ! -f "$UNSAFE_FILE" ]]; then
  echo "A local AdvBench/JailbreakBench unsafe prompt file is required for the 200 unsafe rows." >&2
  echo "Pass it as the second argument or set UNSAFE_FILE=/path/to/advbench.csv." >&2
  exit 1
fi

if [[ "$VENV_DIR" == "$ROOT_DIR/.venv" && ! -f "$VENV_DIR/bin/activate" && ! -f "$VENV_DIR/Scripts/activate" && -f "$ROOT_DIR/venv/bin/activate" ]]; then
  VENV_DIR="$ROOT_DIR/venv"
fi

if [[ ! -f "$VENV_DIR/bin/activate" && ! -f "$VENV_DIR/Scripts/activate" ]]; then
  echo "Missing virtualenv at $VENV_DIR. Run scripts/setup_ssh_env.sh first or set VENV_DIR." >&2
  exit 1
fi

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
fi

export PYTHONBREAKPOINT=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$(dirname "$DATASET_PATH")" "$OUTPUT_DIR" "$LOG_DIR"

quant_flags=()
if [[ "$QUANTIZATION" == "8bit" ]]; then
  quant_flags+=(--load-8bit)
elif [[ "$QUANTIZATION" == "4bit" ]]; then
  quant_flags+=(--load-4bit)
fi

prepare_cmd=(
  python prepare_multimodal_gpo_dataset.py
  --unsafe-file "$UNSAFE_FILE"
  --unsafe-source-name advbench
  --max-safe "$MAX_SAFE"
  --max-unsafe "$MAX_UNSAFE"
  --seed "$SEED"
  --image-path "$IMAGE_PATH"
  --output "$DATASET_PATH"
)

if [[ -n "$UNSAFE_FIELD" ]]; then
  prepare_cmd+=(--unsafe-field "$UNSAFE_FIELD")
fi

if [[ -n "$SAFE_FILE" ]]; then
  if [[ ! -f "$SAFE_FILE" ]]; then
    echo "Safe prompt file not found: $SAFE_FILE" >&2
    exit 1
  fi
  prepare_cmd+=(--safe-file "$SAFE_FILE" --safe-source-name safe_local)
  if [[ -n "$SAFE_FIELD" ]]; then
    prepare_cmd+=(--safe-field "$SAFE_FIELD")
  fi
fi

echo "Preparing 400-row multimodal GPO-V dataset at $DATASET_PATH"
"${prepare_cmd[@]}"

read -r -a step_args <<< "$PROBE_STEPS"

probe_cmd=(
  python run_gpo_v_lladav_probe.py
  --gpo-v-root "$GPO_V_ROOT"
  --input-jsonl "$DATASET_PATH"
  --output-dir "$OUTPUT_DIR"
  --device-map "$DEVICE_MAP"
  --dtype "$DTYPE"
  --probe-steps "${step_args[@]}"
  --all-layers
  --attack-steps "$ATTACK_STEPS"
  --generation-steps "$GENERATION_STEPS"
  --gen-length "$GEN_LENGTH"
  --block-length "$BLOCK_LENGTH"
  --save-every "$SAVE_EVERY"
  --write-final-checkpoint
)

if [[ "$RESUME" == "1" ]]; then
  probe_cmd+=(--resume)
fi

probe_cmd+=("${quant_flags[@]}")

printf '%q ' "${probe_cmd[@]}" > "$LOG_DIR/last_probe_command.txt"
printf '\n' >> "$LOG_DIR/last_probe_command.txt"

echo "Running full multimodal GPO-V probe attack."
echo "Output: $OUTPUT_DIR"
"${probe_cmd[@]}"

python -m scripts.verify_full_multimodal_probe_run \
  --checkpoint-dir "$OUTPUT_DIR" \
  --expected-count "$((MAX_SAFE + MAX_UNSAFE))" \
  --expected-safe-count "$MAX_SAFE" \
  --expected-unsafe-count "$MAX_UNSAFE" \
  --expected-unsafe-attack-steps "$ATTACK_STEPS" \
  --expected-hidden-dim 4096 \
  --probe-steps "${step_args[@]}" \
  --layers $(seq 0 31) \
  --report-json "$LOG_DIR/verification_summary.json"

echo "Full multimodal probe attack run finished."
echo "Final artifacts are in $ROOT_DIR/$OUTPUT_DIR"
echo "Verification summary is in $ROOT_DIR/$LOG_DIR/verification_summary.json"
