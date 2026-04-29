#!/bin/bash

PICKLE_DIR="pickles"

echo "Running cot_run qwen on bigbench_movie part 4..."

python3 main.py \
    --pickle_path $PICKLE_DIR/bigbench_movie.pkl \
    --dataset bigbench_movie \
    --model qwen \
    --thinking \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 1 \
    --confidence \
    --vanilla_only \
    --temperature 0.9 \
    --debug_conf \
    --tag cots-sup-429-p4-qwen-bm-8 \
    --experimental_extra_box \
    --nb_cot_samples 8 \
    --sample_range 190 251 \
    --discord
