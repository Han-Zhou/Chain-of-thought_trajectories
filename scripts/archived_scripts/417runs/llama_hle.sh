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
    --tag 417-llama-hle-100-defaultjk \
    --nb_dropout_samples 10 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --sample_size 100 \
    --discord \
    --save_cots
