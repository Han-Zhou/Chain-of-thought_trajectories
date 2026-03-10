#!/bin/bash

srun \
    --job-name=trial_cot_trajectories \
    --qos=high \
    --partition=compute \
    --nodes=1 \
    --gres=gpu:1 \
    --time=2:00:00 \
    --mem=128G \
    --cpus-per-task=24 \
    bash -c "
        set -euo pipefail
        module load cuda/12.4
        module load conda
        conda activate cot
        ./actual_run.sh
    "

    # --nodelist=lux-2-node-22 \

# srun \
#     --job-name=trial_cot_trajectories \
#     --qos=high \
#     --partition=compute \
#     --nodes=1 \
#     --gres=gpu:4 \
#     --time=2:00:00 \
#     --mem=128G \
#     --cpus-per-task=48 \
#     bash -c "
#         set -euo pipefail
#         module load cuda/12.4
#         module load conda
#         conda activate cot
#         python3 temp.py
#     "

