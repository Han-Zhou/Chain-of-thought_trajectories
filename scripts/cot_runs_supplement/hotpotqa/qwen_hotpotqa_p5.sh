#!/bin/bash

PICKLE_DIR="pickles"

echo "Running cot_run qwen on hotpotqa p5..."

python3 main.py \
    --pickle_path $PICKLE_DIR/hotpotqa.pkl \
    --dataset hotpotqa \
    --model qwen \
    --thinking \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 1 \
    --confidence \
    --vanilla_only \
    --temperature 0.9 \
    --debug_conf \
    --tag cots-sup-425-p5-qwen-hotpotqa-8 \
    --experimental_extra_box \
    --nb_cot_samples 8 \
    --sample_range 415 501 \
    --discord
