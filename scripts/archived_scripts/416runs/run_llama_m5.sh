#!/bin/bash
#SBATCH --job-name=run-llama-m5-full
#SBATCH --qos=high
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=4-8:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

#SBATCH --output=/storage/backup/han/backup_workspace/cot-eval/.slurm_logs/%j_%x.out
#SBATCH --error=/storage/backup/han/backup_workspace/cot-eval/.slurm_logs/%j_%x.err


notify_discord() {
  local message="$1"
  curl -s -H "Content-Type: application/json" \
    -d "{\"content\": \"$message\"}" \
    "$DISCORD_WEBHOOK" > /dev/null 2>&1
}

trap 'notify_discord "❌ **llama-m5-full** failed on $(hostname) (Job $SLURM_JOB_ID) at line $LINENO with exit code $?"' ERR

set -euo pipefail

module load cuda/12.4
module load conda
conda activate cot

./scripts/416runs/llama_m5.sh

notify_discord "✅ **llama-m5-full** completed successfully (Job $SLURM_JOB_ID)"
