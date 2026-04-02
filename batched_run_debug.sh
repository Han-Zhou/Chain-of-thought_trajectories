#!/bin/bash

echo "Starting run ..."

python3 main_debug.py \
    --dataset bigbench_movie \
    --pickle_path /storage/backup/han/backup_workspace/cot-eval/llama_8b_31_bigbench_movie_recommendation.pkl \
    --model llama \
    --shot_mode zero \
    --confidence \
    --debug_conf \
    --tag debug-0402-top-10-llama \
    --nb_dropout_samples 10 \
    --experimental_jackknife \
    --discord
