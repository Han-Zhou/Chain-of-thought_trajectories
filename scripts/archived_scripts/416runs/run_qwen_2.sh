#!/bin/bash
#SBATCH --job-name=run-qwen-2
#SBATCH --qos=high
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
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

trap 'notify_discord "❌ **qwen-2 ** failed on $(hostname) (Job $SLURM_JOB_ID) at line $LINENO with exit code $?"' ERR

set -euo pipefail

module load cuda/12.4
module load conda
conda activate cot

./scripts/416runs/qwen_2.sh

notify_discord "✅ **qwen-2** completed successfully (Job $SLURM_JOB_ID)"
