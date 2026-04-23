#!/bin/bash

echo "Running cot_run qwen on cs1qa_type..."

PICKLE_DIR="pickles"

python3 main.py \
    --pickle_path $PICKLE_DIR/cs1qa_type.pkl \
    --dataset cs1qa_type \
    --model qwen \
    --thinking \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 1 \
    --confidence \
    --vanilla_only \
    --temperature 0.9 \
    --debug_conf \
    --tag cots-422-qwen-cs1qa_type-8 \
    --experimental_extra_box \
    --nb_cot_samples 8 \
    --discord


