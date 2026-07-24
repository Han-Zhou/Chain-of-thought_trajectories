#!/bin/bash

TRAJ_DIR="trajectories/1_full_trajectories/cots-420-llama-bm-32_llama_regular_bigbench_movie_zero_0_250_type2_conf_vanillaonly_mod-none_samples32/"
OUT_DIR="trajectories/2_vanilla_llama_patch/PATCHED-llama-bm-32/"

echo "Patching llama bigbench_movie vanilla confidence scores..."
echo "  trajectory_dir: $TRAJ_DIR"
echo "  output_dir    : $OUT_DIR"

python3 patch_llama_cots.py \
    "$TRAJ_DIR" \
    --output_dir "$OUT_DIR" \
    --dataset bigbench_movie \
    --model llama \
    --shot_mode zero \
    --type 2 \
    --discord
