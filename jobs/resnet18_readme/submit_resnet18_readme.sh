#!/usr/bin/env bash

# Submit the Cartesian grid split across one-GPU and two-GPU PBS arrays.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$PROJECT_DIR"

PBS_FILES=(jobs/resnet18_readme/resnet18_readme.pbs)
GPU_COUNTS=(1)
TASK_COUNTS=(8)
EXPERIMENT_COUNTS=(8)
EXPERIMENT_PATH=experiments/resnet18_readme
DRY_RUN="${DRY_RUN:-0}"

EXP_NAME="${1:-}"
if [[ -z "$EXP_NAME" ]]; then
  read -r -p 'Experiment name: ' EXP_NAME
fi
if [[ ! "$EXP_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Experiment name must contain only letters, numbers, dots, underscores, or hyphens" >&2
  exit 2
fi

LOG_ROOT="$PROJECT_DIR/jobs/resnet18_readme/logs/$EXP_NAME"
mkdir -p "$LOG_ROOT"

for PBS_FILE in "${PBS_FILES[@]}"; do
  if [[ ! -f "$PBS_FILE" ]]; then
    echo "PBS file does not exist: $PBS_FILE" >&2
    exit 2
  fi
done

for INDEX in "${!PBS_FILES[@]}"; do
  PBS_FILE="${PBS_FILES[$INDEX]}"
  GPUS="${GPU_COUNTS[$INDEX]}"
  TASKS="${TASK_COUNTS[$INDEX]}"
  EXPERIMENTS="${EXPERIMENT_COUNTS[$INDEX]}"
  JOB_NAME="${EXP_NAME}_${GPUS}g"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY RUN: qsub -N %q -v %q -o %q %q\n' \
      "${JOB_NAME:0:15}" "EXP_NAME=$EXP_NAME" "$LOG_ROOT" "$PBS_FILE"
    printf '  %s task(s), %s GPU(s) each, %s experiment(s)\n' \
      "$TASKS" "$GPUS" "$EXPERIMENTS"
    continue
  fi

  JOB_ID="$(qsub -N "${JOB_NAME:0:15}" -v "EXP_NAME=$EXP_NAME" \
    -o "$LOG_ROOT" "$PBS_FILE")"
  ARRAY_JOB_TAG="${JOB_ID%%.*}"
  if [[ "$ARRAY_JOB_TAG" =~ ^[0-9]+$ ]]; then
    ARRAY_JOB_TAG="${ARRAY_JOB_TAG}[]"
  fi
  printf 'Submitted %s experiment(s) as %s task(s) with %s GPU(s): %s\n' \
    "$EXPERIMENTS" "$TASKS" "$GPUS" "$JOB_ID"
  printf 'Remote result folder: %s/%s/%s\n' \
    "$EXPERIMENT_PATH" "$EXP_NAME" "$ARRAY_JOB_TAG"
  printf 'Download these results later with: %s %q %q\n' \
    jobs/resnet18_readme/download_resnet18_readme.sh "$EXP_NAME" "$ARRAY_JOB_TAG"
done
