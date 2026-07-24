#!/bin/bash

PICKLE_DIR="pickles"

echo "Running coinflip llama on pregenerated bigbench_movie..."


for cot_sample_idx in {8..15}; do
    python3 main.py \
        --from_pregenerated /storage/backup/han/backup_workspace/cot-eval/trajectories/1_full_trajectories/cots-420-llama-bm-32_llama_regular_bigbench_movie_zero_0_250_type2_conf_vanillaonly_mod-none_samples32 \
        --dataset bigbench_movie \
        --model llama \
        --shot_mode zero \
        --max_new_tokens 15000 \
        --type 2 \
        --confidence \
        --debug_conf \
        --tag test-426-2-pre-llama-coinflip-100 \
        --nb_dropout_samples 100 \
        --cot_sample_idx "$cot_sample_idx" \
        --experimental_extra_box \
        --experimental_preserve_final_step \
        --discord
done



