"""Dispatch original MetaMix-guided JS/JS or feature-MSE SmallCNN training."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--objective', choices=['metamix', 'feature_mse'], required=True)
p.add_argument('--severity-upperbound', type=float, required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--classifier-consistency', choices=['infonce', 'js'], default='infonce')
p.add_argument('--classifier-lr', type=float)
p.add_argument('--mixer-lr', type=float)
p.add_argument('--batch-size', type=int, default=32)
p.add_argument('--momentum', type=float)
p.add_argument('--scheduler', choices=['none', 'cosine'])
p.add_argument('--scheduler-period', type=int)
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
runner = 'metamix_jointbn.py' if a.objective == 'metamix' else 'chain_aware_mixing.py'
command = [sys.executable, '-u', str(ROOT/'reproduce'/runner),
           '--device', 'cuda', '--backbone', 'smallcnn', '--epochs-per-task', '10',
           '--seed', str(a.seed), '--severity-upperbound', str(a.severity_upperbound),
           '--root', 'experiments/metamix_cifar10/data', '--output', str(a.output_dir)]
command += ['--batch-size', str(a.batch_size)]
if a.objective == 'feature_mse':
    command += ['--consistency', a.classifier_consistency,
                '--outer-consistency', 'feature_match']
overrides = ['classifier_lr', 'mixer_lr', 'momentum', 'scheduler', 'scheduler_period']
for key in overrides:
    value = getattr(a, key)
    if value is not None:
        command += ['--' + key.replace('_', '-'), str(value)]
subprocess.run(command, cwd=ROOT, check=True)
