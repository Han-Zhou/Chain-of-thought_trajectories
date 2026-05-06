#!/bin/bash

PICKLE_DIR="pickles"

echo "Running true bootstrap qwen on pregenerated codeqa..."


python3 main.py \
    --from_pregenerated /storage/backup/han/backup_workspace/cot-eval/trajectories/1_full_trajectories/MERGED-cots-503-qwen-codeqa-8 \
    --dataset codeqa \
    --model qwen \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 1 \
    --confidence \
    --debug_conf \
    --tag bootstrap-codeqa-505-qwen \
    --nb_dropout_samples 100 \
    --experimental_extra_box \
    --experimental_preserve_final_step \
    --experimental_bootstrap \
    --discord


