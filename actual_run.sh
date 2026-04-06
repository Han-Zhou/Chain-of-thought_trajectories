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
    --model llama \
    --sample_size 5 \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --type 2 \
    --tag debug-llama

python3 main.py \
    --dataset logiqa \
    --model llama \
    --sample_size 5 \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --confidence \
    --type 2 \
    --debug \
    --debug_conf \
    --tag debug-llama \
    --nb_dropout_samples 10 \
    --experimental_jackknife

    # --debug \


