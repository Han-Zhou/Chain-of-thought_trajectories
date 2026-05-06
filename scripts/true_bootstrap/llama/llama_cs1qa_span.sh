#!/bin/bash

PICKLE_DIR="pickles"

echo "Running true bootstrap llama on pregenerated cs1qa_span..."


python3 main.py \
    --from_pregenerated /storage/backup/han/backup_workspace/cot-eval/trajectories/1_full_trajectories/cots-427-llama-cs1qa_span-32_llama_regular_cs1qa_span_zero_0_500_type2_conf_vanillaonly_mod-none_samples32 \
    --dataset cs1qa_span \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag bootstrap-cs1qa-span-505-llama \
    --nb_dropout_samples 100 \
    --experimental_extra_box \
    --experimental_preserve_final_step \
    --experimental_bootstrap \
    --discord


