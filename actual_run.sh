#!/bin/bash

echo "Starting run ..."

# python3 main.py \
#     --dataset logiqa \
#     --model qwen \
#     --sample_size 10 \
#     --shot_mode few


python3 main.py \
    --dataset logiqa \
    --model llama \
    --sample_size 1 \
    --shot_mode zero 
    # --max_new_tokens 4096




