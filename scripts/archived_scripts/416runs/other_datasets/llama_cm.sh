#!/bin/bash

PICKLE_DIR="pickles"

echo "Running llama on college_math (100 samples)..."

python3 main.py \
    --pickle_path $PICKLE_DIR/college_math.pkl \
    --dataset college_math \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag 416-llama-cm-100-defaultjk \
    --nb_dropout_samples 10 \
    --sample_size 100 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord \
    --save_cots
