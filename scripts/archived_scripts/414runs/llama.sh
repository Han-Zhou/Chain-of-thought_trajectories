#!/bin/bash

PICKLE_DIR="pickles"

echo "Running llama on both math500 and bigbench..."

# python3 main.py \
#     --pickle_path $PICKLE_DIR/math500.pkl \
#     --dataset math500 \
#     --model llama \
#     --shot_mode zero \
#     --max_new_tokens 20000 \
#     --type 2 \
#     --confidence \
#     --debug_conf \
#     --tag 414-llama-math500 \
#     --nb_dropout_samples 100 \
#     --experimental_jackknife default \
#     --experimental_extra_box \
#     --discord

python3 main.py \
    --pickle_path $PICKLE_DIR/bigbench_movie.pkl \
    --dataset bigbench_movie \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag 414-llama-bigbench \
    --nb_dropout_samples 100 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord




