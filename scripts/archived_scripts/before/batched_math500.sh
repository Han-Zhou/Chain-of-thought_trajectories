#!/bin/bash

PICKLE_DIR="pickles"

echo "Running math500..."

python3 main.py \
    --pickle_path $PICKLE_DIR/math500.pkl \
    --dataset math500 \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag debug-0402-math500 \
    --nb_dropout_samples 10 \
    --experimental_jackknife \
    --discord
