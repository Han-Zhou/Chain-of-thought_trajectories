#!/bin/bash

PICKLE_DIR="pickles"

echo "Running qwen on bfcl (100 samples)..."

python3 main.py \
    --pickle_path $PICKLE_DIR/bfcl.pkl \
    --dataset bfcl \
    --model qwen \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 1 \
    --confidence \
    --debug_conf \
    --tag 417-qwen-bfcl-100-defaultjk \
    --nb_dropout_samples 10 \
    --sample_size 100 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord \
    --save_cots \
    --thinking
