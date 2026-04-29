#!/bin/bash

PICKLE_DIR="pickles"

echo "Running fake bootstrap llama on pregenerated logiqa..."


python3 main.py \
    --from_pregenerated /storage/backup/han/backup_workspace/cot-eval/trajectories/1_full_trajectories/cots-420-llama-logiqa-32_llama_regular_logiqa_zero_0_500_type2_conf_vanillaonly_mod-none_samples32 \
    --dataset logiqa \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag fb-llama-429-logiqa \
    --nb_dropout_samples 100 \
    --experimental_extra_box \
    --experimental_preserve_final_step \
    --experimental_fake_bootstrap_hook \
    --discord
