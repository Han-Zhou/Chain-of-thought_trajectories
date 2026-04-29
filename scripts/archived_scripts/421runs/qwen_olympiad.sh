#!/bin/bash

PICKLE_DIR="pickles"

echo "Running qwen on olympiadbench (100 samples)..."

python3 main.py \
    --pickle_path $PICKLE_DIR/olympiadbench.pkl \
    --dataset olympiadbench \
    --model qwen \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 1 \
    --confidence \
    --debug_conf \
    --tag 421-qwen-olympiad-rest100 \
    --nb_dropout_samples 10 \
    --sample_range 69 100 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord \
    --thinking
