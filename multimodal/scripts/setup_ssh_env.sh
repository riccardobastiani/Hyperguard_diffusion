#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPO_V_ROOT="${1:-${GPO_V_ROOT:-}}"

cd "$ROOT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing Python. Set PYTHON_BIN=/path/to/python3 or install python3." >&2
  exit 1
fi

if [[ -n "$GPO_V_ROOT" ]]; then
  if [[ ! -f "$GPO_V_ROOT/llava/model/builder.py" ]]; then
    echo "GPO_V_ROOT must point to the upstream LLaDA-V train folder." >&2
    echo "Expected file: $GPO_V_ROOT/llava/model/builder.py" >&2
    exit 1
  fi
  printf 'GPO_V_ROOT=%q\n' "$GPO_V_ROOT" > "$ROOT_DIR/.env.multimodal"
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
else
  echo "Virtualenv was created, but no activate script was found under $VENV_DIR." >&2
  exit 1
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

python - <<'PY'
import importlib.util

required = ["torch", "transformers", "datasets", "numpy", "sklearn", "geoopt"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing required packages after install: {missing}")

import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

if [[ -z "$GPO_V_ROOT" ]]; then
  echo "Environment ready. Pass the GPO-V LLaDA-V path to the smoke script:"
  echo "  bash scripts/run_multimodal_smoke.sh /path/to/GPO-V/LLaDA-V"
else
  echo "Environment ready. Saved GPO_V_ROOT to .env.multimodal"
fi
