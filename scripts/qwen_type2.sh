#!/bin/bash

PICKLE_DIR="pickles"

echo "Running qwen type 2 on bigbench_movie..."

python3 main.py \
    --pickle_path $PICKLE_DIR/bigbench_movie.pkl \
    --dataset bigbench_movie \
    --model qwen \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag 406-qwen-type2 \
    --nb_dropout_samples 10 \
    --experimental_jackknife \
    --discord
