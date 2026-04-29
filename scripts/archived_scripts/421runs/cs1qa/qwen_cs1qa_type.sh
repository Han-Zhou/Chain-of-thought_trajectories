#!/bin/bash

PICKLE_DIR="pickles"

echo "Running qwen on cs1qa_type subset..."

python3 main.py \
    --pickle_path $PICKLE_DIR/cs1qa_type.pkl \
    --dataset cs1qa_type \
    --model qwen \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 1 \
    --confidence \
    --debug_conf \
    --tag 421-qwen-cs1qa-type-subset \
    --nb_dropout_samples 10 \
    --sample_size 100 \
    --experimental_jackknife default \
    --experimental_extra_box \
    --discord \
    --thinking
