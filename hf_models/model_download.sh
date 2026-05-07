#!/bin/bash
# victim models: dLLMs
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ./LLaDA-8B-Instruct

# refine prompt model
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./Qwen2.5-7B-Instruct

# evaluator--ASR-e
huggingface-cli download qylu4156/strongreject-15k-v1 --local-dir ./strongreject-15k-v1