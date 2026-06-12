#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_URL="${GPO_V_REPO_ZIP_URL:-https://anonymous.4open.science/api/repo/GPO-V-0250/zip}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-$ROOT_DIR/external/GPO-V-0250}"
ZIP_PATH="${GPO_V_ZIP:-$EXTERNAL_ROOT/GPO-V-0250.zip}"
TMP_DIR="$EXTERNAL_ROOT/.extract_tmp"
LLADA_V_ROOT="$EXTERNAL_ROOT/LLaDA-V"
LAVIDA_ROOT="$EXTERNAL_ROOT/LaViDA"
BUILDER_PATH="$LLADA_V_ROOT/llava/model/builder.py"

find_cmd() {
  if [[ -x /usr/bin/find ]]; then
    echo /usr/bin/find
  elif command -v gfind >/dev/null 2>&1; then
    command -v gfind
  elif command -v find >/dev/null 2>&1; then
    command -v find
  else
    echo "Missing find command." >&2
    exit 1
  fi
}

download_zip() {
  mkdir -p "$EXTERNAL_ROOT"

  if [[ -n "${GPO_V_ZIP:-}" ]]; then
    if [[ ! -f "$GPO_V_ZIP" ]]; then
      echo "GPO_V_ZIP was set but the file does not exist: $GPO_V_ZIP" >&2
      exit 1
    fi
    echo "Using local GPO-V zip: $GPO_V_ZIP"
    return
  fi

  echo "Downloading GPO-V upstream zip:"
  echo "  $REPO_URL"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --output "$ZIP_PATH" "$REPO_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$ZIP_PATH" "$REPO_URL"
  else
    echo "Missing downloader: install curl or wget, or set GPO_V_ZIP=/path/to/GPO-V-0250.zip." >&2
    exit 1
  fi
}

extract_zip() {
  if ! command -v unzip >/dev/null 2>&1; then
    echo "Missing unzip. Install unzip, or extract the zip manually under $EXTERNAL_ROOT." >&2
    exit 1
  fi

  rm -rf "$TMP_DIR"
  mkdir -p "$TMP_DIR"
  unzip -q "$ZIP_PATH" -d "$TMP_DIR"

  local extracted_llada
  local extracted_lavida
  local finder
  finder="$(find_cmd)"
  extracted_llada="$("$finder" "$TMP_DIR" -type f -path '*/LLaDA-V/llava/model/builder.py' -print -quit)"
  extracted_lavida="$("$finder" "$TMP_DIR" -type d -path '*/LaViDA' -print -quit)"

  if [[ -z "$extracted_llada" ]]; then
    echo "Downloaded archive did not contain LLaDA-V/llava/model/builder.py." >&2
    exit 1
  fi

  rm -rf "$LLADA_V_ROOT"
  mkdir -p "$EXTERNAL_ROOT"
  mv "$(dirname "$(dirname "$(dirname "$extracted_llada")")")" "$LLADA_V_ROOT"

  if [[ -n "$extracted_lavida" ]]; then
    rm -rf "$LAVIDA_ROOT"
    mv "$extracted_lavida" "$LAVIDA_ROOT"
  fi

  rm -rf "$TMP_DIR"
  if [[ "${KEEP_GPO_V_ZIP:-0}" != "1" && -z "${GPO_V_ZIP:-}" ]]; then
    rm -f "$ZIP_PATH"
  fi
}

write_env_file() {
  local rel_root="external/GPO-V-0250/LLaDA-V"
  printf 'GPO_V_ROOT=%q\n' "$rel_root" > "$ROOT_DIR/.env.multimodal"
}

main() {
  if [[ -f "$BUILDER_PATH" && "${FORCE_GPO_V_DOWNLOAD:-0}" != "1" ]]; then
    echo "GPO-V upstream already present:"
    echo "  $BUILDER_PATH"
  else
    download_zip
    extract_zip
  fi

  if [[ ! -f "$BUILDER_PATH" ]]; then
    echo "GPO-V setup failed. Missing required file:" >&2
    echo "  $BUILDER_PATH" >&2
    exit 1
  fi

  write_env_file
  echo "GPO-V upstream ready."
  echo "GPO_V_ROOT=external/GPO-V-0250/LLaDA-V"
  echo "Verified builder:"
  echo "  $BUILDER_PATH"
}

main "$@"
