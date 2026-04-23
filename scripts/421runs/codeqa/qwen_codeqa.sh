#!/bin/bash

PICKLE_DIR="pickles"

echo "Running qwen on codeqa (subset)..."

python3 main.py \
    --pickle_path $PICKLE_DIR/codeqa.pkl \
    --dataset codeqa \
    --model qwen \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 1 \
    --confidence \
    --debug_conf \
    --tag 421-qwen-codeqa-subset \
    --nb_dropout_samples 10 \
    --sample_size 100 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord \
    --thinking
