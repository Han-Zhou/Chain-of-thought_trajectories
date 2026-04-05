#!/bin/bash

PICKLE_DIR="pickles"

echo "Running hotpotqa..."

python3 main.py \
    --pickle_path $PICKLE_DIR/hotpotqa.pkl \
    --dataset hotpotqa \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag debug-0402-hotpotqa \
    --nb_dropout_samples 10 \
    --experimental_jackknife \
    --discord
