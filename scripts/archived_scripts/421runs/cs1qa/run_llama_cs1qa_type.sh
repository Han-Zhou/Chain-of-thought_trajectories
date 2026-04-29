#!/bin/bash
#SBATCH --job-name=run-llama-cs1qa-type-subset
#SBATCH --qos=high
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
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

trap 'notify_discord "❌ **llama-cs1qa-type-subset** failed on $(hostname) (Job $SLURM_JOB_ID) at line $LINENO with exit code $?"' ERR

set -euo pipefail

module load cuda/12.4
module load conda
conda activate cot

./scripts/421runs/cs1qa/llama_cs1qa_type.sh

notify_discord "✅ **llama-cs1qa-type-subset** completed successfully (Job $SLURM_JOB_ID)"
