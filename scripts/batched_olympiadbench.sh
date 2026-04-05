#!/bin/bash

PICKLE_DIR="pickles"

echo "Running olympiadbench..."

python3 main.py \
    --pickle_path $PICKLE_DIR/olympiadbench.pkl \
    --dataset olympiadbench \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag debug-0402-olympiadbench \
    --nb_dropout_samples 10 \
    --experimental_jackknife \
    --discord
