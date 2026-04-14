#!/bin/bash

PICKLE_DIR="pickles"

echo "Running llama on bigbench_movie (no-cots)..."

python3 main.py \
    --pickle_path $PICKLE_DIR/bigbench_movie.pkl \
    --dataset bigbench_movie \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 5000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag 411_llama-bigbench-no-cots-type2 \
    --nb_dropout_samples 10 \
    --experimental_modification no-cots \
    --experimental_extra_box \
    --discord
