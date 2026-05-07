#!/bin/bash
# victim models: dLLMs
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ./LLaDA-8B-Instruct

# refine prompt model

# evaluator--ASR-e
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ./Meta-Llama-3-8B-Instruct