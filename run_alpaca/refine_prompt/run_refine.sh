#!/bin/bash
set -euo pipefail

cd /workspace/Hyperguard_diffusion/run_alpaca/refine_prompt

PYTHON_BIN=${PYTHON_BIN:-/workspace/Hyperguard_diffusion/.venv/bin/python}
version=${1:-v1}
hf_model_path="/workspace/Hyperguard_diffusion/hf_models/Qwen2.5-7B-Instruct"

# Optional API mode. Leave empty to use local Qwen.
api_model_name=""
api_key=""
base_url=""

prompt_template_path="/workspace/Hyperguard_diffusion/run_alpaca/refine_prompt/redteam_prompt_template.txt"
attack_prompt="/workspace/Hyperguard_diffusion/run_alpaca/refine_prompt/alpaca_subset_689_seed42.csv"
output_json="/workspace/Hyperguard_diffusion/run_alpaca/refine_prompt/alpaca_data_refined_${version}.json"
max_new_tokens=200

"${PYTHON_BIN}" main.py \
  --hf_model_path "${hf_model_path}" \
  --api_model_name "${api_model_name}" \
  --prompt_template_path "${prompt_template_path}" \
  --attack_prompt "${attack_prompt}" \
  --output_json "${output_json}" \
  --base_url "${base_url}" \
  --api_key "${api_key}" \
  --max_new_tokens "${max_new_tokens}"
