#!/bin/bash

echo "Starting run ..."


# python3 main.py \
#     --dataset logiqa \
#     --model llama \
#     --sample_size 1 \
#     --shot_mode zero 
    # --max_new_tokens 4096


python3 main.py \
    --dataset logiqa \
    --model gpt \
    --sample_size 1 \
    --shot_mode few \
    --thinking
    # --max_new_tokens 4096


