#!/bin/bash

PICKLE_DIR="pickles"

echo "Running logiqa..."

python3 main.py \
    --pickle_path $PICKLE_DIR/logiqa.pkl \
    --dataset logiqa \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag debug-0402-logiqa \
    --nb_dropout_samples 10 \
    --experimental_jackknife \
    --discord
