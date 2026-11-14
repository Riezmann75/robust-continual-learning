#!/usr/bin/env bash
set -euo pipefail
cd /home/riezmann/dev/robust-continual-learning
export CUDA_VISIBLE_DEVICES=0
PYTHON=/home/riezmann/miniconda3/envs/liver_transplant/bin/python
for recipe in metamix:js feature_mse:infonce feature_mse:js; do
  objective="${recipe%:*}"
  classifier="${recipe#*:}"
  for severity in 3 5 7; do
    for seed in 17 28 39; do
      name="${objective}_${classifier}_sev${severity}_seed${seed}"
      echo "Starting $name at $(date -Is)"
      "$PYTHON" -u scripts/train_mixer_severity.py \
        --objective "$objective" \
        --classifier-consistency "$classifier" \
        --severity-upperbound "$severity" --seed "$seed" \
        --batch-size 128 --classifier-lr 0.012 --mixer-lr 0.02 \
        --momentum 0.9 --scheduler cosine --scheduler-period 10 \
        --output-dir "experiments/local_bs128_cosine/$name" \
        >> "logs/bs128_cosine/$name.log" 2>&1
      echo "Completed $name at $(date -Is)"
    done
  done
done
