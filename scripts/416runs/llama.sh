#!/bin/bash

PICKLE_DIR="pickles"

echo "Running llama on both math500 and bigbench..."


CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --pickle_path $PICKLE_DIR/bigbench_movie.pkl \
    --dataset bigbench_movie \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag 416-half-fake-multi-llama-bbm \
    --nb_dropout_samples 10 \
    --experimental_jackknife default \
    --experimental_grid \
    --experimental_extra_box \
    --discord \
    --temperature 0.9 \
    --nb_cot_samples 10 \
    --sample_range 150 251 \
    --save_cots &

CUDA_VISIBLE_DEVICES=1 python3 main.py \
    --pickle_path $PICKLE_DIR/bigbench_movie.pkl \
    --dataset bigbench_movie \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag 416-half-true-multi-llama-bbm \
    --nb_dropout_samples 10 \
    --experimental_jackknife true \
    --experimental_grid \
    --experimental_extra_box \
    --discord \
    --temperature 0.9 \
    --nb_cot_samples 10 \
    --sample_range 150 251 \
    --save_cots &

wait



