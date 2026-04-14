#!/bin/bash
#SBATCH --job-name=debug_olympiadbench
#SBATCH --qos=high
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=4-8:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

#SBATCH --output=/storage/backup/han/backup_workspace/cot-eval/.slurm_logs/%x_%j.out
#SBATCH --error=/storage/backup/han/backup_workspace/cot-eval/.slurm_logs/%x_%j.err

DISCORD_WEBHOOK="https://discord.com/api/webhooks/1487181261226381433/YwHw1Te8ScdalSfBIJCj1G7r9-8Ubbw1tcRODk-0phcILmoVK1jxUUB8DnXJQpwTSgGS"

notify_discord() {
  local message="$1"
  curl -s -H "Content-Type: application/json" \
    -d "{\"content\": \"$message\"}" \
    "$DISCORD_WEBHOOK" > /dev/null 2>&1
}

trap 'notify_discord "❌ **olympiadbench** failed on $(hostname) (Job $SLURM_JOB_ID) at line $LINENO with exit code $?"' ERR

set -euo pipefail

module load cuda/12.4
module load conda
conda activate cot

./scripts/batched_olympiadbench.sh

notify_discord "✅ **olympiadbench** completed successfully (Job $SLURM_JOB_ID)"
