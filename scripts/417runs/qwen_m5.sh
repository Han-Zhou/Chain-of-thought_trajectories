#!/bin/bash

PICKLE_DIR="pickles"

echo "Running qwen on both math500..."


python3 main.py \
    --pickle_path $PICKLE_DIR/bigbench_movie.pkl \
    --dataset bigbench_movie \
    --model qwen \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 1 \
    --confidence \
    --debug_conf \
    --tag 417-qwen-m5-full-defaultjk \
    --nb_dropout_samples 100 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord \
    --save_cots  \
    --thinking
    



