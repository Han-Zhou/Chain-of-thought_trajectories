#!/bin/bash

set -euo pipefail

PICKLE_DIR="pickles"

echo "Running sequential vLLM generation for Qwen on math500..."

python3 main.py \
    --pickle_path "$PICKLE_DIR/math500.pkl" \
    --dataset math500 \
    --model qwen \
    --backend vllm \
    --vllm_max_model_len 32768 \
    --vllm_gpu_memory_utilization 0.92 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_num_seqs 1 \
    --vllm_language_model_only \
    --thinking \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 1 \
    --temperature 0.9 \
    --tag vllm-sequential-cots-723-qwen-m500-32 \
    --experimental_extra_box \
    --nb_cot_samples 24 \
    --sample_size 3 \
    --discord
