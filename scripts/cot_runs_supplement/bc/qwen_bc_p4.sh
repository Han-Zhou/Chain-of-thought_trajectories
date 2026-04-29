#!/bin/bash

PICKLE_DIR="pickles"

echo "Running cot_run qwen on bigbench_causal part 4..."

python3 main.py \
    --pickle_path $PICKLE_DIR/bigbench_causal.pkl \
    --dataset bigbench_causal \
    --model qwen \
    --thinking \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 1 \
    --confidence \
    --vanilla_only \
    --temperature 0.9 \
    --debug_conf \
    --tag cots-sup-429-p4-qwen-bc-8 \
    --experimental_extra_box \
    --nb_cot_samples 8 \
    --sample_range 113 150 \
    --discord
