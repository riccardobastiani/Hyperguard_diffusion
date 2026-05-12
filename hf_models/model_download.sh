#!/bin/bash

# ── PHASE 1: Attack + Refinement ──────────────────────────────────────────────
# victim model (dLLM)
# huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ./LLaDA-8B-Instruct

# refine prompt model
# huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./Qwen2.5-7B-Instruct

# ── PHASE 2: Evaluation (ASR-e) ───────────────────────────────────────────────
# After refinement is complete, free space by running:
#   rm -rf ./Qwen2.5-7B-Instruct
# Then download the judge model:
  huggingface-cli download meta-llama/Llama-Guard-3-8B --local-dir ./Llama-Guard-3-8B --exclude "original/*"