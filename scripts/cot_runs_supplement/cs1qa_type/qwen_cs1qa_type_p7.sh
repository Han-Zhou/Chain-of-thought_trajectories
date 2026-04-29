#!/bin/bash

PICKLE_DIR="pickles"

echo "Running cot_run qwen on cs1qa type p7..."

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
    --tag cots-sup-428-qwen-cs1qa_type_p7-8 \
    --experimental_extra_box \
    --nb_cot_samples 8 \
    --sample_range 433 501 \
    --discord

