#!/bin/bash

TRAJ_DIR="trajectories/1_full_trajectories/MERGED-cots-430-qwen-bm-8/"

# echo "Patching qwen bigbench_movie vanilla confidence scores in-place..."
# echo "  trajectory_dir: $TRAJ_DIR"

python3 patch_qwen_cots.py \
    "$TRAJ_DIR" \
    --dataset bigbench_movie \
    --model qwen \
    --thinking \
    --shot_mode zero \
    --type 1 \
    --experimental_extra_box \
    --discord


# python3 toy_bootstrap_demo.py \
#     "${TRAJ_DIR}traj_0_sample_0.json" \
#     --out toy_bootstrap_demo.json
