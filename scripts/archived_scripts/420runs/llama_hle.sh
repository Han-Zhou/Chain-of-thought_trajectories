#!/bin/bash

PICKLE_DIR="pickles"

echo "Running llama on hle..."

python3 main.py \
    --pickle_path $PICKLE_DIR/hle.pkl \
    --dataset hle \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 1 \
    --confidence \
    --debug_conf \
    --tag 420-llama-hle-50-defaultjk \
    --nb_dropout_samples 10 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --sample_size 50 \
    --discord \
    --save_cots
