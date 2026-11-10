#!/usr/bin/env bash

# Download every grid point belonging to one PBS array job.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
REMOTE_HOST="${REMOTE_HOST:-hpc}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/home/users/ntu/giabaoca/myProject-12004268/giabao/robust-continual-learning}"
EXPERIMENT_PATH=experiments/prediction_match_js_smallcnn
LOG_PATH=jobs/prediction_js/logs

EXP_NAME="${1:-}"
if [[ -z "$EXP_NAME" ]]; then
  read -r -p 'Experiment name: ' EXP_NAME
fi
if [[ ! "$EXP_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Experiment name must contain only letters, numbers, dots, underscores, or hyphens" >&2
  exit 2
fi

ARRAY_JOB_ID="${2:-}"
if [[ -z "$ARRAY_JOB_ID" ]]; then
  read -r -p 'Array job ID (for example 15020042[]): ' ARRAY_JOB_ID
fi
ARRAY_JOB_TAG="${ARRAY_JOB_ID%%.*}"
if [[ ! "$ARRAY_JOB_TAG" =~ ^[0-9]+\[\]$ ]]; then
  echo "Array job ID must look like 15020042[] or 15020042[].server" >&2
  exit 2
fi

RSYNC_ARGS=(-avh --partial --progress --exclude=.DS_Store)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  RSYNC_ARGS+=(--dry-run --itemize-changes)
fi

REMOTE_EXPERIMENT="$REMOTE_PROJECT/$EXPERIMENT_PATH/$EXP_NAME/$ARRAY_JOB_TAG/"
LOCAL_EXPERIMENT="$PROJECT_DIR/$EXPERIMENT_PATH/$EXP_NAME/$ARRAY_JOB_TAG/"
REMOTE_LOGS="$REMOTE_PROJECT/$LOG_PATH/$EXP_NAME/"
LOCAL_LOGS="$PROJECT_DIR/$LOG_PATH/$EXP_NAME/"
mkdir -p "$LOCAL_EXPERIMENT" "$LOCAL_LOGS"

rsync "${RSYNC_ARGS[@]}" \
  "$REMOTE_HOST:$REMOTE_EXPERIMENT" \
  "$LOCAL_EXPERIMENT"

rsync "${RSYNC_ARGS[@]}" \
  "$REMOTE_HOST:$REMOTE_LOGS" \
  "$LOCAL_LOGS"

printf 'Downloaded all grid results to: %s
' "$LOCAL_EXPERIMENT"
printf 'Downloaded logs to:             %s
' "$LOCAL_LOGS"
