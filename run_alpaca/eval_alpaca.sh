#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <attack_method> <defense_method> <model_name> <version>"
    exit 1
fi

attack_method=$1
defense_method=$2
model_name=$3
version=$4
PYTHON_BIN=${PYTHON_BIN:-"${ROOT_DIR}/.venv/bin/python"}

together_api_key=""
judge_model_path="${ROOT_DIR}/hf_models/Llama-Guard-3-8B"
use_local_judge=true

if command -v nvidia-smi >/dev/null 2>&1; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{min=999999; idx=0} {if ($1 < min) {min=$1; idx=NR-1}} END{print idx}')
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

cd "${ROOT_DIR}/run_alpaca"

if [[ "$model_name" == *"llada_instruct"* ]]; then
    model_path="${ROOT_DIR}/hf_models/LLaDA-8B-Instruct"
    python_script="models/alpaca_llada.py"
    steps=128
    gen_length=128
    mask_id=126336
    mask_counts=36
elif [[ "$model_name" == *"llada_1.5"* ]]; then
    model_path="${ROOT_DIR}/hf_models/LLaDA-1.5"
    python_script="models/alpaca_llada.py"
    steps=128
    gen_length=128
    mask_id=126336
    mask_counts=36
else
    echo "Unknown or unsupported Alpaca model name: $model_name"
    echo "Supported examples: llada_instruct, llada_1.5"
    exit 1
fi

attack_prompt="${ROOT_DIR}/run_alpaca/refine_prompt/alpaca_data_refined_${version}.json"
output_json="${ROOT_DIR}/run_alpaca/attack_results/${model_name}_${attack_method}_attack_${defense_method}_defense_${version}.json"
save_path="${ROOT_DIR}/run_alpaca/eval_results/eval_results_${model_name}_${attack_method}_attack_${defense_method}_defense_${version}.json"

#echo "Running Alpaca model inference with ${python_script}..."
#"${PYTHON_BIN}" ${python_script} \
#    --model_path "${model_path}" \
#    --attack_prompt "${attack_prompt}" \
#    --output_json "${output_json}" \
#    --steps ${steps} \
#    --gen_length ${gen_length} \
#    --mask_id ${mask_id} \
#    --mask_counts ${mask_counts} \
#    --attack_method "${attack_method}" \
#    --defense_method "${defense_method}"

echo "Running AdvBench ASR-e judge on Alpaca outputs..."
if [ "$use_local_judge" = true ]; then
    "${PYTHON_BIN}" eval_metric/evaluate_completions_asr_e.py \
        --data-path "${output_json}" \
        --local \
        --model-path "${judge_model_path}" \
        --output-path "${save_path}"
else
    "${PYTHON_BIN}" eval_metric/evaluate_completions_asr_e.py \
        --data-path "${output_json}" \
        --api-key "${together_api_key}" \
        --output-path "${save_path}"
fi

echo "Running AdvBench ASR-k judge on Alpaca outputs..."
"${PYTHON_BIN}" eval_metric/evaluate_completions_asr_k.py \
    --json_path "${save_path}"

echo "Alpaca DIJA pipeline completed."
