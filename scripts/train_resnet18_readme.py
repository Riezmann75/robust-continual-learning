"""PBS adapter for the eight settings in the README's main comparison."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = (
    'metamix_jointbn', 'matched_random_js', 'matched_random_infonce',
    'matched_learned_infonce', 'matched_infonce_js', 'chain_aware_mixing',
    'chain_aware_js', 'chain_aware_infonce_infonce',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=SETTINGS, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('experiments/metamix_cifar10/data'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    command = [sys.executable, str(ROOT / 'scripts/run_experiment.py'),
               str(ROOT / 'configs' / (args.setting + '.json')),
               '--backbone', 'resnet18', '--device', 'cuda',
               '--data-root', str(args.data_root.resolve()),
               '--output', str(args.output_dir.resolve())]
    if not args.dry_run:
        command.append('--run')
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
