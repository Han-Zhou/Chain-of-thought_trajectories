#!/bin/bash

PICKLE_DIR="pickles"

echo "Running true bootstrap qwen on pregenerated bigbench_movie p5..."


python3 main.py \
    --from_pregenerated /storage/backup/han/backup_workspace/cot-eval/trajectories/1_full_trajectories/MERGED-cots-422-qwen-bm-8_qwen_thinking_bigbench_movie_zero_0_250_type1_conf_vanillaonly_mod-none_samples8 \
    --dataset bigbench_movie \
    --model qwen \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 1 \
    --confidence \
    --debug_conf \
    --tag bootstrap-bm-428-qwen-p5 \
    --nb_dropout_samples 100 \
    --experimental_extra_box \
    --experimental_preserve_final_step \
    --experimental_bootstrap \
    --sample_range 200 250 \
    --discord

