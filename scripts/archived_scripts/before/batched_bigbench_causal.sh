#!/bin/bash

PICKLE_DIR="pickles"

echo "Running bigbench_causal..."

python3 main.py \
    --pickle_path $PICKLE_DIR/bigbench_causal.pkl \
    --dataset bigbench_causal \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag debug-0402-bigbench-causal \
    --nb_dropout_samples 10 \
    --experimental_jackknife \
    --discord
