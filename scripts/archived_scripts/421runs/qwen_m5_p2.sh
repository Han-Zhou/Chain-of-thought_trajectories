#!/bin/bash

PICKLE_DIR="pickles"

echo "Running qwen on math500 part 2..."



python3 main.py \
    --pickle_path $PICKLE_DIR/math500.pkl \
    --dataset math500 \
    --model qwen \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 1 \
    --confidence \
    --debug_conf \
    --tag 421-qwen-m5-full-p2 \
    --nb_dropout_samples 100 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord \
    --sample_range 256 500 \
    --thinking


    



