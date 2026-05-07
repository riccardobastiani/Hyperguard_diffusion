#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <attack_method> <defense_method> <model_name> <version>"
    exit 1
fi

# Assign input parameters
attack_method=$1
defense_method=$2
model_name=$3
version=$4
together_api_key=""  # TODO: set your Together API key here (used only if not using local judge)
judge_model_path="/workspace/Hyperguard_diffusion/hf_models/Meta-Llama-3-8B-Instruct"  # TODO: set path to local judge model
use_local_judge=true  # Set to false to use Together API instead

API_KEY="" # TODO: Set your OpenAI API key here
BASE_URL="" # TODO: Set your OpenAI API base URL here if needed

# Automatically select the GPU with the least memory used
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{min=999999; idx=0} {if ($1 < min) {min=$1; idx=NR-1}} END{print idx}')
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Navigate to working directory
cd /workspace/Hyperguard_diffusion/run_jailbreakbench || { echo "Failed to change directory"; exit 1; }

# Define paths based on model name
if [[ "$model_name" == *"llada_instruct"* ]]; then
    model_path="/workspace/Hyperguard_diffusion/hf_models/LLaDA-8B-Instruct"  # TODO: Update this path
    python_script="models/jailbreakbench_llada.py"
    steps=128
    gen_length=128
    mask_id=126336
    mask_counts=36
elif [[ "$model_name" == *"llada_1.5"* ]]; then
    model_path="/workspace/Hyperguard_diffusion/hf_models/LLaDA-1.5" # TODO: Update this path
    python_script="models/jailbreakbench_llada.py"
    steps=128
    gen_length=128
    mask_id=126336
    mask_counts=36
elif [[ "$model_name" == *"dream_instruct"* ]]; then
    model_path="/workspace/Hyperguard_diffusion/hf_models/Dream-v0-Instruct-7B"  # TODO: Update this path
    python_script="models/jailbreakbench_dream.py"
    steps=64
    gen_length=64
    mask_id=151666
    mask_counts=36
elif [[ "$model_name" == *"dream_coder_instruct"* ]]; then
    model_path="/workspace/Hyperguard_diffusion/hf_models/Dream-Coder-v0-Instruct-7B"  # TODO: Update this path
    python_script="models/jailbreakbench_dream.py"
    steps=64
    gen_length=64
    mask_id=151666
    mask_counts=36
elif [[ "$model_name" == *"dreamon_instruct"* ]]; then
    model_path="/workspace/Hyperguard_diffusion/hf_models/DreamOn-v0-7B"  # TODO: Update this path
    python_script="models/jailbreakbench_dream.py"
    steps=64
    gen_length=64
    mask_id=151666
    mask_counts=36
elif [[ "$model_name" == *"diffucoder_instruct"* ]]; then
    model_path="/workspace/Hyperguard_diffusion/hf_models/DiffuCoder-7B-Instruct"  # TODO: Update this path
    python_script="models/jailbreakbench_dream.py"
    steps=64
    gen_length=64
    mask_id=151666
    mask_counts=36
elif [[ "$model_name" == *"diffucoder_cpgrpo"* ]]; then
    model_path="/workspace/Hyperguard_diffusion/hf_models/DiffuCoder-7B-cpGRPO"  # TODO: Update this path
    python_script="models/jailbreakbench_dream.py"
    steps=64
    gen_length=64
    mask_id=151666
    mask_counts=36
elif [[ "$model_name" == *"mmada_mixcot"* ]]; then
    model_path="/workspace/Hyperguard_diffusion/hf_models/MMaDA-8B-MixCoT"  # TODO: Update this path
    python_script="models/jailbreakbench_mmada.py"
    steps=128
    gen_length=128
    mask_id=126336
    mask_counts=36
else
    echo "Unknown or unsupported model name: $model_name"
    exit 1
fi

attack_prompt="/workspace/Hyperguard_diffusion/run_jailbreakbench/refine_prompt/jailbreakbench_data_refined_${version}.json"
output_json="/workspace/Hyperguard_diffusion/run_jailbreakbench/attack_results/${model_name}_${attack_method}_attack_${defense_method}_defense_${version}.json"


save_path="/workspace/Hyperguard_diffusion/run_jailbreakbench/eval_results/eval_results_${model_name}_${attack_method}_attack_${defense_method}_defense_${version}.json"

# Run ASR-e evaluation
echo "Running JailbreakBench ASR-e..."
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
echo "Running JailbreakBench ASR-k..."
# Use ASR-e output if available, otherwise fall back to raw attack results
if [ -f "${save_path}" ]; then
    asr_k_input="${save_path}"
else
    echo "ASR-e output not found, running ASR-k on raw attack results instead."
    asr_k_input="${output_json}"
fi
python eval_metric/evaluate_completions_asr_k.py \
    --json_path "${asr_k_input}" \


echo "All steps completed successfully."