#!/bin/bash

echo "Starting run ..."


# python3 main.py \
#     --dataset bfcl \
#     --pickle_path pickles/bfcl.pkl \
#     --model llama \
#     --shot_mode zero \
#     --confidence \
#     --debug_conf \
#     --type 1 \
#     --max_new_tokens 20000 \
#     --tag debug-temp-0408-bfcl-llama \
#     --nb_dropout_samples 10 \
#     --experimental_jackknife \
#     --sample_size 1
#     # --experimental_extra_box



PICKLE_DIR="pickles"

echo "Running qwen type 1 on bigbench_movie..."

p    --pickle_path $PICKLE_DIR/bigbench_movie.pkl \
    --dataset bigbench_movie \
    --model qwen \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --type 1 \
    --confidence \
    --thinking \
    --debug_conf \
    --tag debug-413_qwen-boxed-type1 \
    --nb_dropout_samples 10 \
    --experimental_jackknife \
    --experimental_extra_box \
    --sample_range 1 12
    # --sample_size 10

