#!/bin/bash

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
together_api_key=""  # TODO: set your Together API key here (used only if not using local judge)
judge_model_path="${ROOT_DIR}/hf_models/Llama-Guard-3-8B"
use_local_judge=true  # Set to false to use Together API instead

API_KEY=""   # TODO: Set your OpenAI API key here if needed
BASE_URL=""  # TODO: Set your OpenAI API base URL here if needed

# Automatically select the GPU with the least memory used
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{min=999999; idx=0} {if ($1 < min) {min=$1; idx=NR-1}} END{print idx}')
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Navigate to working directory
cd "${ROOT_DIR}/run_advbench" || { echo "Failed to change directory"; exit 1; }

# Define paths based on model name
if [[ "$model_name" == *"llada_instruct"* ]]; then
    model_path="${ROOT_DIR}/hf_models/LLaDA-8B-Instruct"
    python_script="models/advbench_llada.py"
    steps=128
    gen_length=128
    mask_id=126336
    mask_counts=36
elif [[ "$model_name" == *"llada_1.5"* ]]; then
    model_path="${ROOT_DIR}/hf_models/LLaDA-1.5"
    python_script="models/advbench_llada.py"
    steps=128
    gen_length=128
    mask_id=126336
    mask_counts=36
elif [[ "$model_name" == *"dream_instruct"* ]]; then
    model_path="${ROOT_DIR}/hf_models/Dream-v0-Instruct-7B"
    python_script="models/advbench_dream.py"
    steps=64
    gen_length=64
    mask_id=151666
    mask_counts=36
elif [[ "$model_name" == *"dream_coder_instruct"* ]]; then
    model_path="${ROOT_DIR}/hf_models/Dream-Coder-v0-Instruct-7B"
    python_script="models/advbench_dream.py"
    steps=64
    gen_length=64
    mask_id=151666
    mask_counts=36
elif [[ "$model_name" == *"dreamon_instruct"* ]]; then
    model_path="${ROOT_DIR}/hf_models/DreamOn-v0-7B"
    python_script="models/advbench_dream.py"
    steps=64
    gen_length=64
    mask_id=151666
    mask_counts=36
elif [[ "$model_name" == *"diffucoder_instruct"* ]]; then
    model_path="${ROOT_DIR}/hf_models/DiffuCoder-7B-Instruct"
    python_script="models/advbench_dream.py"
    steps=64
    gen_length=64
    mask_id=151666
    mask_counts=36
elif [[ "$model_name" == *"diffucoder_cpgrpo"* ]]; then
    model_path="${ROOT_DIR}/hf_models/DiffuCoder-7B-cpGRPO"
    python_script="models/advbench_dream.py"
    steps=64
    gen_length=64
    mask_id=151666
    mask_counts=36
elif [[ "$model_name" == *"mmada_mixcot"* ]]; then
    model_path="${ROOT_DIR}/hf_models/MMaDA-8B-MixCoT"
    python_script="models/advbench_mmada.py"
    steps=128
    gen_length=128
    mask_id=126336
    mask_counts=36
else
    echo "Unknown or unsupported model name: $model_name"
    exit 1
fi

attack_prompt="${ROOT_DIR}/run_advbench/refine_prompt/advbench_data_refined_${version}.json"
output_json="${ROOT_DIR}/run_advbench/attack_results/${model_name}_${attack_method}_attack_${defense_method}_defense_${version}.json"
save_path="${ROOT_DIR}/run_advbench/eval_results/eval_results_${model_name}_${attack_method}_attack_${defense_method}_defense_${version}.json"

# Run the jailbreak attack
#echo "Running model inference with ${python_script}..."
#python ${python_script} \
#    --model_path "${model_path}" \
#    --attack_prompt "${attack_prompt}" \
#    --output_json "${output_json}" \
#    --steps ${steps} \
#    --gen_length ${gen_length} \
#    --mask_id ${mask_id} \
#    --mask_counts ${mask_counts} \
#    --attack_method "${attack_method}" \
#    --defense_method "${defense_method}"

# Run ASR-e evaluation
echo "Running AdvBench ASR-e..."
if [ "$use_local_judge" = true ]; then
    python eval_metric/evaluate_completions_asr_e.py \
        --data-path "${output_json}" \
        --local \
        --model-path "${judge_model_path}" \
        --output-path "${save_path}"
else
    python eval_metric/evaluate_completions_asr_e.py \
        --data-path "${output_json}" \
        --api-key "${together_api_key}" \
        --output-path "${save_path}"
fi

# Run ASR-k evaluation
echo "Running AdvBench ASR-k..."
if [ -f "${save_path}" ]; then
    asr_k_input="${save_path}"
else
    echo "ASR-e output not found, running ASR-k on raw attack results instead."
    asr_k_input="${output_json}"
fi
python eval_metric/evaluate_completions_asr_k.py \
    --json_path "${asr_k_input}"

echo "All steps completed successfully."
