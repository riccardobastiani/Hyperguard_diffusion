#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}/run_advbench/refine_prompt"

version=$1
hf_model_path="${ROOT_DIR}/hf_models/Qwen2.5-7B-Instruct"

# TODO: Set your API key and base URL if using an API model
api_model_name=""
api_key=""
base_url=""

prompt_template_path="${ROOT_DIR}/run_advbench/refine_prompt/redteam_prompt_template.txt"
attack_prompt="${ROOT_DIR}/run_advbench/refine_prompt/advbench_harmful_behaviors.csv"
output_json="${ROOT_DIR}/run_advbench/refine_prompt/advbench_data_refined_${version}.json"
max_new_tokens=200

python main.py \
  --hf_model_path "${hf_model_path}" \
  --api_model_name "${api_model_name}" \
  --prompt_template_path "${prompt_template_path}" \
  --attack_prompt "${attack_prompt}" \
  --output_json "${output_json}" \
  --base_url "${base_url}" \
  --api_key "${api_key}" \
  --max_new_tokens "${max_new_tokens}"
