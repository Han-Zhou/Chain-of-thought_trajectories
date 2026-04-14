#!/bin/bash

PICKLE_DIR="pickles"

echo "Running college_math..."

python3 main.py \
    --pickle_path $PICKLE_DIR/college_math.pkl \
    --dataset college_math \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag debug-0402-college-math \
    --nb_dropout_samples 10 \
    --experimental_jackknife \
    --discord
