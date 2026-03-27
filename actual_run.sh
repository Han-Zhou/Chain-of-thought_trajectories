#!/bin/bash

echo "Starting run ..."

# python3 main.py \
#     --dataset logiqa \
#     --model qwen \
#     --sample_size 1 \
#     --shot_mode few \
#     --thinking \
#     --max_new_tokens 7000 \
#     --type 1

python3 main.py \
    --dataset logiqa \
    --model qwen \
    --sample_size 1 \
    --shot_mode few \
    --thinking \
    --max_new_tokens 20000 \
    --type 2

# python3 main.py \
#     --dataset logiqa \
#     --model qwen \
#     --sample_size 1 \
#     --shot_mode few \
#     --thinking \
#     --max_new_tokens 20000 \
#     --confidence \
#     --type 2 \
#     --debug \
#     --debug_conf

    # --debug \


