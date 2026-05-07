#!/bin/bash

# # =======================
# # Slurm SBATCH Directives
# # =======================
# #SBATCH --job-name=logiqa_0_299_cot_trajectories
# #SBATCH --qos=high
# #SBATCH --partition=compute
# #SBATCH --nodes=1
# #SBATCH --gres=gpu:2
# #--time=5-00:00:00
# #SBATCH --time=4-8:00:00
# #SBATCH --mem=128G
# #SBATCH --cpus-per-task=24
# #SBATCH --qos=high
# # SBATCH --nodelist=lux-2-node-23

# #SBATCH --output=/storage/backup/han/backup_workspace/cot-eval/.slurm_logs/%x_%j.out
# #SBATCH --error=/storage/backup/han/backup_workspace/cot-eval/.slurm_logs/%x_%j.err
# # =======================

# DISCORD_WEBHOOK="https://discord.com/api/webhooks/1487181261226381433/YwHw1Te8ScdalSfBIJCj1G7r9-8Ubbw1tcRODk-0phcILmoVK1jxUUB8DnXJQpwTSgGS"

# notify_discord() {
#   local message="$1"
#   curl -s -H "Content-Type: application/json" \
#     -d "{\"content\": \"$message\"}" \
#     "$DISCORD_WEBHOOK" > /dev/null 2>&1
# }

# trap 'notify_discord "❌ **run1.sh** failed on $(hostname) (Job $SLURM_JOB_ID) at line $LINENO with exit code $?"' ERR

# set -euo pipefail

# module load cuda/12.4
# module load conda
# conda activate cot

# ./batched_run1.sh

# notify_discord "✅ **run1.sh** completed successfully (Job $SLURM_JOB_ID)"


# sbatch scripts/cot_runs_supplement/codeqa/run_qwen_codeqa_p1.sh
# sbatch scripts/cot_runs_supplement/codeqa/run_qwen_codeqa_p2.sh
# sbatch scripts/cot_runs_supplement/codeqa/run_qwen_codeqa_p3.sh
# sbatch scripts/cot_runs_supplement/codeqa/run_qwen_codeqa_p4.sh
# sbatch scripts/cot_runs_supplement/codeqa/run_qwen_codeqa_p5.sh




# sbatch scripts/cot_runs_supplement/bfcl/run_qwen_bfcl_p1.sh
# sbatch scripts/cot_runs_supplement/bfcl/run_qwen_bfcl_p2.sh
# sbatch scripts/cot_runs_supplement/bfcl/run_qwen_bfcl_p3.sh
# sbatch scripts/cot_runs_supplement/bfcl/run_qwen_bfcl_p4.sh
# sbatch scripts/cot_runs_supplement/bfcl/run_qwen_bfcl_p5.sh




# sbatch scripts/cot_runs/qwen/run_qwen_hotpotqa.sh
# sbatch scripts/cot_runs_supplement/hotpotqa/run_qwen_hotpotqa_p2.sh
# sbatch scripts/cot_runs_supplement/hotpotqa/run_qwen_hotpotqa_p3.sh
# sbatch scripts/cot_runs_supplement/hotpotqa/run_qwen_hotpotqa_p4.sh
# sbatch scripts/cot_runs_supplement/hotpotqa/run_qwen_hotpotqa_p5.sh
# sbatch scripts/cot_runs_supplement/hotpotqa/run_qwen_hotpotqa_p6.sh
# sbatch scripts/cot_runs_supplement/hotpotqa/run_qwen_hotpotqa_p7.sh


# sbatch scripts/cot_runs/qwen/run_qwen_bm.sh
# sbatch scripts/cot_runs_supplement/bm/run_qwen_bm_p2.sh
# sbatch scripts/cot_runs_supplement/bm/run_qwen_bm_p3.sh
# sbatch scripts/cot_runs_supplement/bm/run_qwen_bm_p4.sh



# sbatch scripts/cot_runs/qwen/run_qwen_m500.sh
# sbatch scripts/cot_runs_supplement/m500/run_qwen_m500_p1.sh
# sbatch scripts/cot_runs_supplement/m500/run_qwen_m500_p2.sh
# sbatch scripts/cot_runs_supplement/m500/run_qwen_m500_p3.sh
# sbatch scripts/cot_runs_supplement/m500/run_qwen_m500_p4.sh
# sbatch scripts/cot_runs_supplement/m500/run_qwen_m500_p5.sh
# sbatch scripts/cot_runs_supplement/m500/run_qwen_m500_p6.sh
# sbatch scripts/cot_runs_supplement/m500/run_qwen_m500_p7.sh
# sbatch scripts/cot_runs_supplement/m500/run_qwen_m500_p8.sh
# sbatch scripts/cot_runs_supplement/m500/run_qwen_m500_p9.sh
# sbatch scripts/cot_runs_supplement/m500/run_qwen_m500_p10.sh




# sbatch scripts/428runs/run_qwen_m500.sh



