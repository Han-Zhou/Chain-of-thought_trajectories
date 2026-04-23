#!/bin/bash

echo "Running cot_run llama on codeqa..."

PICKLE_DIR="pickles"

python3 main.py \
    --pickle_path $PICKLE_DIR/codeqa.pkl \
    --dataset codeqa \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 2 \
    --confidence \
    --vanilla_only \
    --temperature 0.9 \
    --debug_conf \
    --tag cots-422-llama-codeqa-32 \
    --nb_cot_samples 32 \
    --discord

