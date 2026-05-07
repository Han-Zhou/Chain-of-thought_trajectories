#!/bin/bash
#SBATCH --job-name=test_flash_attn
#SBATCH --output=/storage/backup/han/backup_workspace/cot-eval/.slurm_logs/%j_%x.out
#SBATCH --error=/storage/backup/han/backup_workspace/cot-eval/.slurm_logs/%j_%x.err
#SBATCH --time=00:10:00
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load modules
module load cuda/12.4

# Activate conda environment
# eval "$(conda shell.bash hook)"
module load conda
conda activate cot

# Create logs directory if it doesn't exist

# Enable detailed CUDA error reporting
export CUDA_LAUNCH_BLOCKING=1

# Run the test script
echo "Testing Flash Attention 2 with Qwen3.5..."
python3 temp_flash_attn.py

echo "Test completed at $(date)"
