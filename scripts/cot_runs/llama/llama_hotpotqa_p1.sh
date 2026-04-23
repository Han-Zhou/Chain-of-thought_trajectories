#!/bin/bash

PICKLE_DIR="pickles"

echo "Running cot_run llama on hotpotqa (p1: 0-500)..."

python3 main.py \
    --pickle_path $PICKLE_DIR/hotpotqa.pkl \
    --dataset hotpotqa \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 2 \
    --confidence \
    --vanilla_only \
    --temperature 0.9 \
    --debug_conf \
    --tag cots-421-llama-hotpotqa-32-p1 \
    --experimental_extra_box \
    --nb_cot_samples 32 \
    --sample_range 0 500 \
    --discord
