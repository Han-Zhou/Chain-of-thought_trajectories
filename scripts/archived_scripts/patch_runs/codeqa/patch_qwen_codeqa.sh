#!/bin/bash

TRAJ_DIR="trajectories/1_full_trajectories/MERGED-cots-503-qwen-codeqa-8"



python3 patch_qwen_cots.py \
    "$TRAJ_DIR" \
    --dataset codeqa \
    --model qwen \
    --thinking \
    --shot_mode zero \
    --type 1 \
    --experimental_extra_box \
    --discord

