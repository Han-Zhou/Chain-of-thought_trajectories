#!/bin/bash

PICKLE_DIR="pickles"

echo "Running mini cot_run qwen on math500 with thinking..."

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
    --debug_conf \
    --tag test2-cots-428-qwen-math500-thinking-8 \
    --experimental_extra_box \
    --sample_size 5 \
    --discord
