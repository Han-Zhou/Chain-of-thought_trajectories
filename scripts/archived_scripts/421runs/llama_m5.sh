#!/bin/bash

PICKLE_DIR="pickles"

echo "Running llama on both math500..."


python3 main.py \
    --pickle_path $PICKLE_DIR/math500.pkl \
    --dataset math500 \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag 421-llama-m5-full \
    --nb_dropout_samples 100 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord



