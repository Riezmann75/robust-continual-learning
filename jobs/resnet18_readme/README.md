# ResNet18 README comparison on PBS

Eight settings from the main README's corrected mixing comparison, in table
order. Each uses scratch ResNet18 with native 512-dimensional features, seed 17,
10 epochs per task, five tasks, and the existing CE + DER++ settings. No scientific
hyperparameters are changed beyond the backbone. The optional augmented-CE and
decoder experiments are separate from this eight-row comparison.

## Setup after cloning (Linux NVIDIA GPU / HPC)

From the cloned repository root, with conda available:

```bash
conda create -n survival_analysis python=3.13 -y
conda activate survival_analysis
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python reproduce/download_cifar10_benchmark.py --root experiments/metamix_cifar10/data
```

If `survival_analysis` already exists, activate it instead of recreating it; the
name matches the generator default. Alternatively regenerate with `--conda-env`
to use a dedicated environment. Download once on a network-enabled node and
keep the data in the shared repo filesystem. The downloader verifies both clean
CIFAR-10 and the CIFAR-10-C evaluation data. No dataset is committed.

Inside an allocated GPU session, verify CUDA before full training:

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

The CUDA 12.8 wheels match the repository's tested build and require a compatible
NVIDIA driver. GPU availability is expected on allocated compute nodes.

## Grid and resources

| Index | Config | Classifier consistency | Mixer objective |
| --- | --- | --- | --- |
| 1 | metamix_jointbn | JS | JS |
| 2 | matched_random_js | JS | none |
| 3 | matched_random_infonce | class-aware InfoNCE | none |
| 4 | matched_learned_infonce | class-aware InfoNCE | class-aware InfoNCE |
| 5 | matched_infonce_js | class-aware InfoNCE | JS |
| 6 | chain_aware_mixing | class-aware InfoNCE | JS |
| 7 | chain_aware_js | JS | JS |
| 8 | chain_aware_infonce_infonce | class-aware InfoNCE | class-aware InfoNCE |

One PBS array, indices 1-8, with one independent experiment per task/GPU.
Each task requests queue `ai`, project `12004268`, 1 GPU, 8 CPUs, 64 GB RAM,
and walltime `8:00:00`. Environment: `survival_analysis`. Notifications use the
generator default address `giabao.cao@ntu.edu.sg` (abort, begin, end).
Eight hours is a scheduling default; ResNet18 runtime has not been benchmarked.

## Inspect and submit

Run from the repository root on the HPC after these files are available there:

```bash
DRY_RUN=1 bash jobs/resnet18_readme/submit_resnet18_readme.sh resnet18_seed17
DRY_RUN=1 EXP_NAME=resnet18_seed17 PBS_ARRAY_INDEX=1 bash jobs/resnet18_readme/resnet18_readme.pbs
bash jobs/resnet18_readme/submit_resnet18_readme.sh resnet18_seed17
```

The last command submits training. Generation and local validation do not submit.
Results are written beneath:

```text
experiments/resnet18_readme/<EXP_NAME>/<ARRAY_JOB_ID[]>/<EXP_NAME>_<INDEX>/
```

Each run has its existing config, source hashes, checkpoints, traces and final
evaluation. Output directories are unique across array submissions. Resume a
particular interrupted run by reusing its exact output directory through the
adapter; a fresh submission creates a new array-ID directory.

## Regenerate and download

The source of truth is the existing liver-transplant `generate_hpc_job.py`;
it is invoked directly, not copied here. `arguments.json` stores the source grid.
The adapter translates the generator's `--output-dir` into the repository
launcher's `--output` and selects the existing config.

```bash
HPC_JOB_GENERATOR=/path/to/liver-transplant/generate_hpc_job.py \
REMOTE_PROJECT=/home/users/ntu/giabaoca/myProject-12004268/giabao/robust-continual-learning \
bash jobs/resnet18_readme/regenerate.sh --force
```

`regenerate.sh` must run with Python 3.10+; defaults remain the generator's defaults.
`--force` intentionally replaces the generated files. Its working directory is
the target repository so trainer, output and helper paths stay repo-relative.
The download helper defaults to your HPC checkout path; you can override it:

```bash
REMOTE_PROJECT=/home/users/ntu/giabaoca/myProject-12004268/giabao/robust-continual-learning \
bash jobs/resnet18_readme/download_resnet18_readme.sh resnet18_seed17 '15020042[]'
```

Replace the sample array ID with the one returned by submission. This helper
uses host alias `hpc` and downloads both results and logs. Its dry-run option
still contacts the remote machine.
