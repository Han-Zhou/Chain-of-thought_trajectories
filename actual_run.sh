#!/bin/bash

echo "Starting run ..."

PICKLE_DIR="pickles"

python3 main.py \
    --pickle_path $PICKLE_DIR/bigbench_movie.pkl \
    --dataset bigbench_movie \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 2000 \
    --type 1 \
    --confidence \
    --debug_conf \
    --tag debug_412-gibberish-conf-llama-boxed-type2 \
    --nb_dropout_samples 10 \
    --experimental_jackknife \
    --experimental_modification gibberish \
    --experimental_extra_box \
    --sample_size 10
    # --sample_range 3 4


# python3 main.py \
#     --pickle_path $PICKLE_DIR/bigbench_movie.pkl \
#     --dataset bigbench_movie \
#     --model llama \
#     --shot_mode zero \
#     --max_new_tokens 20000 \
#     --type 2 \
#     --confidence \
#     --debug_conf \
#     --tag debug-409-llama-boxed-type2 \
#     --nb_dropout_samples 10 \
#     --experimental_jackknife \
#     --experimental_extra_box \
#     --sample_size 10
