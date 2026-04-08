#!/bin/bash

PICKLE_DIR="pickles"

echo "Running llama experimental type 1 on bigbench_movie..."

python3 main.py \
    --pickle_path $PICKLE_DIR/bigbench_movie.pkl \
    --dataset bigbench_movie \
    --model llama \
    --sample_size 1 \
    --shot_mode zero \
    --max_new_tokens 20000 \
    --type 1 \
    --confidence \
    --debug_conf \
    --tag 407_experimental-extra-box-type1 \
    --nb_dropout_samples 10 \
    --experimental_jackknife \
    --experimental_extra_box
