#!/bin/bash

# =======================
# Slurm SBATCH Directives
# =======================
#SBATCH --job-name=cot_trajectories
#SBATCH --qos=high
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#--time=5-00:00:00
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=48

#SBATCH --output=/shared_work/han/cot/.slurm_logs/%x_%j.out
#SBATCH --error=/shared_work/han/cot/.slurm_logs/%x_%j.err
# =======================

set -euo pipefail

module load cuda/12.4
module load conda
conda activate cot

./actual_run.sh


