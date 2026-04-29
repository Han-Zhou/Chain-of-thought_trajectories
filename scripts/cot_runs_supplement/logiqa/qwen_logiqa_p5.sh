#!/bin/bash

PICKLE_DIR="pickles"

echo "Running cot_run qwen on logiqa p5..."

python3 main.py \
    --pickle_path $PICKLE_DIR/logiqa.pkl \
    --dataset logiqa \
    --model qwen \
    --thinking \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 1 \
    --confidence \
    --vanilla_only \
    --temperature 0.9 \
    --debug_conf \
    --tag cots-sup-429-p5-qwen-logiqa-8 \
    --experimental_extra_box \
    --nb_cot_samples 8 \
    --sample_range 400 501 \
    --discord
