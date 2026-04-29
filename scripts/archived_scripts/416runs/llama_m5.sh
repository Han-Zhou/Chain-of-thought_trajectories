#!/bin/bash

PICKLE_DIR="pickles"

echo "Running llama on both math500..."


python3 main.py \
    --pickle_path $PICKLE_DIR/bigbench_movie.pkl \
    --dataset bigbench_movie \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag 416-llama-m5-full-defaultjk \
    --nb_dropout_samples 100 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord \
    --save_cots 



