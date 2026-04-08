#!/bin/bash

srun \
    --job-name=trial_conf-extra-box \
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
        ./scripts/actual_runs/extra_box.sh
    "
