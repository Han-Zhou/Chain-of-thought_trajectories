#!/bin/bash

PICKLE_DIR="pickles"

echo "Running cot_run qwen on math500..."

python3 main.py \
    --pickle_path $PICKLE_DIR/math500.pkl \
    --dataset math500 \
    --model qwen \
    --thinking \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 1 \
    --confidence \
    --vanilla_only \
    --temperature 0.9 \
    --debug_conf \
    --tag debug3-cots-723-qwen-m500-32 \
    --experimental_extra_box \
    --nb_cot_samples 32 \
    --sample_size 3 \
    --discord
