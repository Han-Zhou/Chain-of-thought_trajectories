#!/bin/bash

PICKLE_DIR="pickles"

echo "Running llama on math500 (gibberish + default jackknife)..."

python3 main.py \
    --pickle_path $PICKLE_DIR/math500.pkl \
    --dataset math500 \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 5000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag 411_llama-math500-gibberish-defaultJK-type2 \
    --nb_dropout_samples 10 \
    --experimental_modification gibberish \
    --experimental_jackknife \
    --experimental_extra_box \
    --discord
