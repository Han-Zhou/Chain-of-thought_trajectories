#!/bin/bash

# =======================
# Slurm SBATCH Directives
# =======================
#SBATCH --job-name=first_cot_trajectories
#SBATCH --qos=high
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#--time=5-00:00:00
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24
#SBATCH --qos=high

#SBATCH --output=/storage/backup/han/backup_workspace/cot-eval/.slurm_logs/%x_%j.out
#SBATCH --error=/storage/backup/han/backup_workspace/cot-eval/.slurm_logs/%x_%j.err
# =======================

set -euo pipefail

module load cuda/12.4
module load conda
conda activate cot

./batched_run.sh


