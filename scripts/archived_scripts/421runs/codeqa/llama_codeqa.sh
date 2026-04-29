#!/bin/bash

PICKLE_DIR="pickles"

echo "Running llama on both codeqa (subset)..."


python3 main.py \
    --pickle_path $PICKLE_DIR/codeqa.pkl \
    --dataset codeqa \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag 421-llama-codeqa-subset \
    --nb_dropout_samples 10 \
    --sample_size 100 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord



