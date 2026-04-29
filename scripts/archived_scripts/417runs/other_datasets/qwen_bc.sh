#!/bin/bash

PICKLE_DIR="pickles"

echo "Running qwen on bigbench_causal..."

python3 main.py \
    --pickle_path $PICKLE_DIR/bigbench_causal.pkl \
    --dataset bigbench_causal \
    --model qwen \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 1 \
    --confidence \
    --debug_conf \
    --tag 417-qwen-bc-defaultjk \
    --nb_dropout_samples 10 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord \
    --save_cots \
    --thinking
