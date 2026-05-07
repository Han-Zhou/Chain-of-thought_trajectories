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







# sbatch scripts/fake_bootstrap/llama/run_llama_m500.sh
# sbatch scripts/fake_bootstrap/llama/run_llama_bc.sh
# sbatch scripts/fake_bootstrap/llama/run_llama_bfcl.sh
# sbatch scripts/fake_bootstrap/llama/run_llama_logiqa.sh
sbatch scripts/fake_bootstrap/llama/run_llama_codeqa.sh




# sbatch scripts/true_bootstrap/qwen/bm/run_qwen_bm_p2.sh
# sbatch scripts/true_bootstrap/qwen/bm/run_qwen_bm_p3.sh
# sbatch scripts/true_bootstrap/qwen/bm/run_qwen_bm_p4.sh
# sbatch scripts/true_bootstrap/qwen/bm/run_qwen_bm_p5.sh



# sbatch ./scripts/422runs/run_coinflip.sh
# sbatch ./scripts/422runs/run_grid.sh
# sbatch ./scripts/422runs/run_fakejk.sh
# sbatch ./scripts/422runs/run_truejk.sh
# sbatch ./scripts/422runs/run_bootstrap.sh






# sbatch scripts/fake_coinflip/llama/run_llama_bc.sh
# sbatch scripts/fake_coinflip/llama/run_llama_bfcl.sh
# sbatch scripts/fake_coinflip/llama/run_llama_logiqa.sh
# sbatch scripts/fake_coinflip/llama/run_llama_codeqa.sh


# sbatch scripts/true_bootstrap/llama/run_llama_bm.sh
# sbatch scripts/true_bootstrap/llama/run_llama_m500.sh
# sbatch scripts/true_bootstrap/llama/run_llama_bc.sh
# sbatch scripts/true_bootstrap/llama/run_llama_logiqa.sh
# sbatch scripts/true_bootstrap/llama/run_llama_bfcl.sh
# sbatch scripts/true_bootstrap/llama/run_llama_codeqa.sh



# sbatch scripts/427runs/run_llama_bm_p1.sh
# sbatch scripts/427runs/run_llama_bm_p2.sh
# sbatch scripts/427runs/run_llama_bm_p3.sh
# sbatch scripts/427runs/run_llama_bm_p4.sh


# sbatch ./scripts/cot_runs_supplement/run_qwen_hotpotqa_p2.sh
# sbatch ./scripts/cot_runs_supplement/run_qwen_hotpotqa_p3.sh
# sbatch ./scripts/cot_runs_supplement/run_qwen_hotpotqa_p4.sh
# sbatch ./scripts/cot_runs_supplement/run_qwen_hotpotqa_p5.sh

# sbatch ./scripts/426runs/run_llama_bm_p1.sh
# sbatch ./scripts/426runs/run_llama_bm_p2.sh
# sbatch ./scripts/426runs/run_llama_bm_p3.sh
# sbatch ./scripts/426runs/run_llama_bm_p4.sh













