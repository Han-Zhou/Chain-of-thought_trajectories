#!/bin/bash

PICKLE_DIR="pickles"

echo "Running llama on bigbench_movie..."

python3 main.py \
    --pickle_path $PICKLE_DIR/bigbench_movie.pkl \
    --dataset bigbench_movie \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 2000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag 413_llama-bb-true-boxed-type2 \
    --nb_dropout_samples 10 \
    --experimental_jackknife true \
    --experimental_extra_box \
    --discord \
    --experimental_grid

python3 main.py \
    --pickle_path $PICKLE_DIR/bigbench_movie.pkl \
    --dataset bigbench_movie \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 2000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag 413_llama-bb-default-boxed-type2 \
    --nb_dropout_samples 10 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord \
    --experimental_grid

