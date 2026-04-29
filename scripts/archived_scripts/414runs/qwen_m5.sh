#!/bin/bash

PICKLE_DIR="pickles"

echo "Running qwen on math500..."

python3 main.py \
    --pickle_path $PICKLE_DIR/math500.pkl \
    --dataset math500 \
    --model qwen \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --type 1 \
    --confidence \
    --debug_conf \
    --tag 414-qwen-math500 \
    --nb_dropout_samples 10 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord


