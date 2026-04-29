#!/bin/bash

PICKLE_DIR="pickles"

echo "Running qwen on bigbench_movie..."

python3 main.py \
    --pickle_path $PICKLE_DIR/bigbench_movie.pkl \
    --dataset bigbench_movie \
    --model qwen \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 1 \
    --confidence \
    --debug_conf \
    --tag 421-qwen-bm-restfull \
    --nb_dropout_samples 100 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord \
    --sample_range 169 500 \
    --thinking
