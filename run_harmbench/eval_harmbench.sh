#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <attack_method> <defense_method> <model_name> <version>"
    exit 1
fi

attack_method=$1
defense_method=$2
model_name=$3
version=$4

API_KEY="" # TODO: Set your OpenAI API key here
BASE_URL="" # TODO: Set your OpenAI API base URL here if needed

IDLE_GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{if ($1 == 0) print NR-1}' | head -n 1)
export CUDA_VISIBLE_DEVICES=${IDLE_GPU:-0}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


# Navigate to working directory for model inference
cd /workspace/Hyperguard_diffusion/run_harmbench || { echo "Failed to change directory to run_harmbench"; exit 1; }

# Define paths based on model name
if [[ "$model_name" == *"llada_instruct"* ]]; then
    model_path="/workspace/Hyperguard_diffusion/hf_models/LLaDA-8B-Instruct"  # TODO: Update this path
    python_script="models/harmbench_llada.py"
    steps=128
    gen_length=128
    mask_id=126336
    mask_counts=36
elif [[ "$model_name" == *"llada_1.5"* ]]; then
    model_path="/workspace/Hyperguard_diffusion/hf_models/LLaDA-1.5" # TODO: Update this path
    python_script="models/harmbench_llada.py"
    steps=128
    gen_length=128
    mask_id=126336
    mask_counts=36
elif [[ "$model_name" == *"dream_instruct"* ]]; then
    model_path="/workspace/Hyperguard_diffusion/hf_models/Dream-v0-Instruct-7B"  # TODO: Update this path
    python_script="models/harmbench_dream.py"
    steps=64
    gen_length=64
    mask_id=151666
    mask_counts=36
elif [[ "$model_name" == *"dream_coder_instruct"* ]]; then
    model_path="/workspace/Hyperguard_diffusion/hf_models/Dream-Coder-v0-Instruct-7B"  # TODO: Update this path
    python_script="models/harmbench_dream.py"
    steps=64
    gen_length=64
    mask_id=151666
    mask_counts=36
elif [[ "$model_name" == *"dreamon_instruct"* ]]; then
    model_path="/workspace/Hyperguard_diffusion/hf_models/DreamOn-v0-7B"  # TODO: Update this path
    python_script="models/harmbench_dream.py"
    steps=64
    gen_length=64
    mask_id=151666
    mask_counts=36
elif [[ "$model_name" == *"diffucoder_instruct"* ]]; then
    model_path="/workspace/Hyperguard_diffusion/hf_models/DiffuCoder-7B-Instruct"  # TODO: Update this path
    python_script="models/harmbench_dream.py"
    steps=64
    gen_length=64
    mask_id=151666
    mask_counts=36
elif [[ "$model_name" == *"diffucoder_cpgrpo"* ]]; then
    model_path="/workspace/Hyperguard_diffusion/hf_models/DiffuCoder-7B-cpGRPO"  # TODO: Update this path
    python_script="models/harmbench_dream.py"
    steps=64
    gen_length=64
    mask_id=151666
    mask_counts=36
elif [[ "$model_name" == *"mmada_mixcot"* ]]; then
    model_path="/workspace/Hyperguard_diffusion/hf_models/MMaDA-8B-MixCoT"  # TODO: Update this path
    python_script="models/harmbench_mmada.py"
    steps=128
    gen_length=128
    mask_id=126336
    mask_counts=36
else
    echo "Unknown or unsupported model name: $model_name"
    exit 1
fi

# Define prompt and output file paths
attack_prompt="/workspace/Hyperguard_diffusion/run_harmbench/refine_prompt/harmbench_behaviors_text_all_refined_${version}.json"
output_json="/workspace/Hyperguard_diffusion/run_harmbench/attack_results/${model_name}_${attack_method}_attack_${defense_method}_defense_${version}.json"


# TODO: Run the jailbreak attack
echo "Running model inference with ${python_script}..."
python ${python_script} \
  --model_path "${model_path}" \
  --attack_prompt "${attack_prompt}" \
  --output_json "${output_json}" \
  --steps ${steps} \
  --gen_length ${gen_length} \
  --mask_id ${mask_id} \
  --mask_counts ${mask_counts} \
  --attack_method "${attack_method}" \
  --defense_method "${defense_method}"


# Define evaluation paths
cls_path="/workspace/Hyperguard_diffusion/hf_models/Llama-Guard-3-8B"
completions_path="${output_json}"
save_path="/workspace/Hyperguard_diffusion/run_harmbench/eval_results/eval_results_${model_name}_${attack_method}_attack_${defense_method}_defense_${version}.json"

# TODO: Run ASR-e evaluation
echo "Running HarmBench ASR-e..."
python eval_metric/evaluate_completions_asr_e.py \
    --cls_path "${cls_path}" \
    --completions_path "${completions_path}" \
    --save_path "${save_path}" \


# TODO: Run ASR-k evaluation
echo "Running HarmBench ASR-k..."
python eval_metric/evaluate_completions_asr_k.py \
    --json_path "${save_path}" \

# TODO: Run Harmful Score evaluation
echo "Running Harmful Score evaluation..."
harmful_score_save_path="/workspace/Hyperguard_diffusion/run_harmbench/eval_results/harmfulness_score/${model_name}_${attack_method}_${defense_method}_${version}.json"
python eval_metric/harmfulscore.py \
    --input_file "${save_path}" \
    --output_file "${harmful_score_save_path}" \
    --judge_model "gpt-4o" \
    --policy_model "gpt-4o" \
    --num_processes 100 \
    --api_key "${API_KEY}" \
    --base_url "${BASE_URL}"

echo "All steps completed successfully."