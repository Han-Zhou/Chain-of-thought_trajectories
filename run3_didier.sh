# #!/bin/bash

# # =======================
# # Slurm SBATCH Directives
# # =======================
# #SBATCH --job-name=logiqa_300_651_cot_trajectories
# #SBATCH --qos=high
# #SBATCH --partition=compute
# #SBATCH --nodes=1
# #SBATCH --gres=gpu:2
# #--time=5-00:00:00
# #SBATCH --time=4-8:00:00
# #SBATCH --mem=128G
# #SBATCH --cpus-per-task=24
# #SBATCH --qos=high
# # SBATCH --nodelist=lux-2-node-20

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

# trap 'notify_discord "❌ **run2.sh** failed on $(hostname) (Job $SLURM_JOB_ID) at line $LINENO with exit code $?"' ERR

# set -euo pipefail

# module load cuda/12.4
# module load conda
# conda activate cot

# ./batched_run2.sh

# notify_discord "✅ **run2.sh** completed successfully (Job $SLURM_JOB_ID)"







sbatch scripts/cot_runs_supplement/cs1qa_span/run_qwen_cs1qa_span_p5.sh
sbatch scripts/cot_runs_supplement/cs1qa_span/run_qwen_cs1qa_span_p6.sh
sbatch scripts/cot_runs_supplement/cs1qa_span/run_qwen_cs1qa_span_p7.sh
sbatch scripts/cot_runs_supplement/cs1qa_span/run_qwen_cs1qa_span_p8.sh







# sbatch scripts/cot_runs_supplement/logiqa/run_qwen_logiqa_p2.sh
# sbatch scripts/cot_runs_supplement/logiqa/run_qwen_logiqa_p3.sh
# sbatch scripts/cot_runs_supplement/logiqa/run_qwen_logiqa_p4.sh
# sbatch scripts/cot_runs_supplement/logiqa/run_qwen_logiqa_p5.sh





# sbatch scripts/cot_runs_supplement/bc/run_qwen_bc_p2.sh
# sbatch scripts/cot_runs_supplement/bc/run_qwen_bc_p3.sh
# sbatch scripts/cot_runs_supplement/bc/run_qwen_bc_p4.sh
# sbatch scripts/cot_runs_supplement/bc/run_qwen_bc_p5.sh




#  sbatch scripts/cot_runs_supplement/cs1qa_type/run_qwen_cs1qa_type_p2.sh
# sbatch scripts/cot_runs_supplement/cs1qa_type/run_qwen_cs1qa_type_p3.sh
# sbatch scripts/cot_runs_supplement/cs1qa_type/run_qwen_cs1qa_type_p4.sh
# sbatch scripts/cot_runs_supplement/cs1qa_type/run_qwen_cs1qa_type_p5.sh
# sbatch scripts/cot_runs_supplement/cs1qa_type/run_qwen_cs1qa_type_p6.sh
# sbatch scripts/cot_runs_supplement/cs1qa_type/run_qwen_cs1qa_type_p7.sh


# sbatch scripts/cot_runs_supplement/cs1qa_span/run_qwen_cs1qa_span_p2.sh
# sbatch scripts/cot_runs_supplement/cs1qa_span/run_qwen_cs1qa_span_p3.sh
# sbatch scripts/cot_runs_supplement/cs1qa_span/run_qwen_cs1qa_span_p4.sh
# sbatch scripts/cot_runs_supplement/cs1qa_span/run_qwen_cs1qa_span_p5.sh
# sbatch scripts/cot_runs_supplement/cs1qa_span/run_qwen_cs1qa_span_p6.sh
# sbatch scripts/cot_runs_supplement/cs1qa_span/run_qwen_cs1qa_span_p7.sh
# sbatch scripts/cot_runs_supplement/cs1qa_span/run_qwen_cs1qa_span_p8.sh
# sbatch scripts/cot_runs_supplement/cs1qa_span/run_qwen_cs1qa_span_p9.sh

