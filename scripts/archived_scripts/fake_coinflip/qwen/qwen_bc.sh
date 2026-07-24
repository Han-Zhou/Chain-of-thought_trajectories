#!/bin/bash

PICKLE_DIR="pickles"

echo "Running coinflip qwen on pregenerated bigbench_causal..."


python3 main.py \
    --from_pregenerated /storage/backup/han/backup_workspace/cot-eval/trajectories/1_full_trajectories/cots-422-qwen-bc-8_qwen_thinking_bigbench_causal_zero_0_187_type1_conf_vanillaonly_mod-none_samples8 \
    --model qwen \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag coinflip-bc-427-qwen \
    --nb_dropout_samples 100 \
    --experimental_extra_box \
    --experimental_preserve_final_step \
    --discord
