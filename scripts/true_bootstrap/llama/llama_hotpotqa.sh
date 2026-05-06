#!/bin/bash

PICKLE_DIR="pickles"

echo "Running true bootstrap llama on pregenerated hotpotqa..."


python3 main.py \
    --from_pregenerated /storage/backup/han/backup_workspace/cot-eval/trajectories/1_full_trajectories/cots-427-llama-hotpotqa-32_llama_regular_hotpotqa_zero_0_500_type2_conf_vanillaonly_mod-none_samples32 \
    --dataset hotpotqa \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag bootstrap-hotpotqa-505-llama \
    --nb_dropout_samples 100 \
    --experimental_extra_box \
    --experimental_preserve_final_step \
    --experimental_bootstrap \
    --discord


