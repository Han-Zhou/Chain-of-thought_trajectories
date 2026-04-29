#!/bin/bash

PICKLE_DIR="pickles"

echo "Running llama on math500..."

python3 main.py \
    --pickle_path $PICKLE_DIR/math500.pkl \
    --dataset math500 \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 2000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag 413_llama-m5-true-boxed-type2 \
    --nb_dropout_samples 10 \
    --experimental_jackknife true \
    --experimental_extra_box \
    --discord \
    --experimental_grid

python3 main.py \
    --pickle_path $PICKLE_DIR/math500.pkl \
    --dataset math500 \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 2000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag 413_llama-m5-default-boxed-type2 \
    --nb_dropout_samples 10 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord \
    --experimental_grid

