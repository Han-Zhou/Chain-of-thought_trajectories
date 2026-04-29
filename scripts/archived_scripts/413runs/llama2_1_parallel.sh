#!/bin/bash

PICKLE_DIR="pickles"

echo "Running llama on bigbench_movie (parallel: true and default jackknife)..."

# Run both jackknife modes in parallel, each on a separate GPU
CUDA_VISIBLE_DEVICES=0 python3 main.py \
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
    --experimental_grid &

CUDA_VISIBLE_DEVICES=1 python3 main.py \
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
    --experimental_grid &

# Wait for both to complete
wait

echo "Both bigbench_movie runs completed"
