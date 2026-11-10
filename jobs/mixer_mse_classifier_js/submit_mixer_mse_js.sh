#!/usr/bin/env bash

# Submit the complete Cartesian grid as one PBS job array.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$PROJECT_DIR"

PBS_FILE=jobs/mixer_mse_classifier_js/mixer_mse_js.pbs
EXPERIMENT_PATH=experiments/mixer_mse_js_bs128_cosine
DRY_RUN="${DRY_RUN:-0}"

EXP_NAME="${1:-}"
if [[ -z "$EXP_NAME" ]]; then
  read -r -p 'Experiment name: ' EXP_NAME
fi
if [[ ! "$EXP_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Experiment name must contain only letters, numbers, dots, underscores, or hyphens" >&2
  exit 2
fi

LOG_ROOT="$PROJECT_DIR/jobs/mixer_mse_classifier_js/logs/$EXP_NAME"

if [[ ! -f "$PBS_FILE" ]]; then
  echo "PBS file does not exist: $PBS_FILE" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT"

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY RUN: qsub -N %q -v %q -o %q %q
' \
    "${EXP_NAME:0:15}" "EXP_NAME=$EXP_NAME" "$LOG_ROOT" "$PBS_FILE"
  printf 'Grid points: 9
'
  exit 0
fi

JOB_ID="$(qsub -N "${EXP_NAME:0:15}" -v "EXP_NAME=$EXP_NAME" -o "$LOG_ROOT" "$PBS_FILE")"
ARRAY_JOB_TAG="${JOB_ID%%.*}"
if [[ "$ARRAY_JOB_TAG" =~ ^[0-9]+$ ]]; then
  ARRAY_JOB_TAG="${ARRAY_JOB_TAG}[]"
fi
printf 'Submitted %s grid points as array job: %s
' 9 "$JOB_ID"
printf 'Remote result folder: %s/%s/%s
' \
  "$EXPERIMENT_PATH" "$EXP_NAME" "$ARRAY_JOB_TAG"
printf 'Download all grid results later with: %s %q %q
' \
  jobs/mixer_mse_classifier_js/download_mixer_mse_js.sh "$EXP_NAME" "$ARRAY_JOB_TAG"
