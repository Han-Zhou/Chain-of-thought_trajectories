#!/bin/bash

PICKLE_DIR="pickles"

echo "Running true coinflip llama on pregenerated math500..."

python3 main.py \
    --from_pregenerated /storage/backup/han/backup_workspace/cot-eval/trajectories/1_full_trajectories/cots-420-llama-m500-32_llama_regular_math500_zero_0_498_type2_conf_vanillaonly_mod-none_samples32 \
    --dataset math500 \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag hollow-m500-427-llama \
    --vanilla_only \
    --experimental_extra_box \
    --experimental_hollow 1 \
    --discord
