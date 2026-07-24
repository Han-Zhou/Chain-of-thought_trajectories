#!/bin/bash

PICKLE_DIR="pickles"

echo "Running true coinflip llama on pregenerated bfcl..."

python3 main.py \
    --from_pregenerated trajectories/1_full_trajectories/cots-420-llama-bfcl-32_llama_regular_bfcl_zero_0_500_type2_conf_vanillaonly_mod-none_samples32 \
    --dataset bfcl \
    --model llama \
    --shot_mode zero \
    --max_new_tokens 15000 \
    --type 2 \
    --confidence \
    --debug_conf \
    --tag hollow-bfcl-427-llama \
    --vanilla_only \
    --experimental_extra_box \
    --experimental_hollow 1 \
    --discord
