#!/bin/bash

PICKLE_DIR="pickles"

echo "Running cot_run llama on hotpotqa..."

python3 main.py \
    --pickle_path $PICKLE_DIR/hotpotqa.pkl \
    --dataset hotpotqa \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 2 \
    --confidence \
    --vanilla_only \
    --temperature 0.9 \
    --debug_conf \
    --tag cots-420-llama-hotpotqa-32 \
    --experimental_extra_box \
    --nb_cot_samples 32 \
    --discord
