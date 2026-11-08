#!/usr/bin/env bash
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
GENERATOR="${HPC_JOB_GENERATOR:-/Users/riezmann/dev/liver-transplant/generate_hpc_job.py}"
python "$GENERATOR" \
  --training-script scripts/train_resnet18_readme.py \
  --arguments-json jobs/resnet18_readme/arguments.json \
  --queue ai \
  --experiment-path experiments/resnet18_readme \
  --job-name resnet18_readme \
  --job-folder jobs/resnet18_readme \
  --remote-project "${REMOTE_PROJECT:-/path/to/robust-continual-learning}" \
  "$@"
