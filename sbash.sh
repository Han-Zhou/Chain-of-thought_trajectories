#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <job-id>" >&2
    exit 2
fi

job_id=$1
step_id="${job_id}.0"

if scontrol show step "$step_id" >/dev/null 2>&1; then
    exec sattach "$step_id"
else
    exec srun --jobid="$job_id" --pty bash
fi
