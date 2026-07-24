#!/bin/bash

PICKLE_DIR="pickles"

echo "Running coinflip qwen on pregenerated logiqa..."


python3 main.py \
    --from_pregenerated trajectories/1_full_trajectories/MERGED-cots-427-qwen-logiqa-8 \
    --dataset logiqa \
    --model qwen \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag coinflip-logiqa-427-qwen \
    --nb_dropout_samples 100 \
    --experimental_extra_box \
    --experimental_preserve_final_step \
    --discord
