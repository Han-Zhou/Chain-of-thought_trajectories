#!/bin/bash

PICKLE_DIR="pickles"

echo "Running qwen on bfcl..."

python3 main.py \
    --pickle_path $PICKLE_DIR/bfcl.pkl \
    --dataset bfcl \
    --model qwen \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --type 1 \
    --confidence \
    --debug_conf \
    --tag 414-qwen-bfcl \
    --nb_dropout_samples 10 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord \
    --sample_size 100 \
    --save_cots




