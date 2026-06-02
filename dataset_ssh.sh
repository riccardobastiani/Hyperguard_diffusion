#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNSAFE_FILE="${UNSAFE_FILE:-$ROOT_DIR/data/advbench.csv}"
SAFE_FILE="${SAFE_FILE:-$ROOT_DIR/data/alpaca_safe.csv}"
IMAGE_PATH="${IMAGE_PATH:-$ROOT_DIR/gpo_v/assets/example_input_image.jpg}"
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/data/multimodal_gpo/llada_v_gpo_prompts.jsonl}"

cd "$ROOT_DIR"

if [[ ! -f "$UNSAFE_FILE" ]]; then
  echo "Unsafe prompt file not found: $UNSAFE_FILE" >&2
  exit 1
fi

if [[ ! -f "$SAFE_FILE" ]]; then
  echo "Safe prompt file not found: $SAFE_FILE" >&2
  exit 1
fi

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "Image file not found: $IMAGE_PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

python3 prepare_multimodal_gpo_dataset.py \
  --unsafe-file "$UNSAFE_FILE" \
  --unsafe-field goal \
  --unsafe-source-name advbench \
  --safe-file "$SAFE_FILE" \
  --safe-field instruction \
  --max-safe 50 \
  --max-unsafe 50 \
  --image-path "$IMAGE_PATH" \
  --output "$OUTPUT_PATH"

echo "Dataset created at: $OUTPUT_PATH"