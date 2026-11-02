# Robust continual learning

Controlled Split-CIFAR10 experiments using a small CNN, DER++ replay, and
clean/augmented-view objectives. Includes completed historical results and a new
**augmented CE** runner with an optional **input-reconstruction decoder**.

**New augmented-CE/decoder variants have passed short correctness tests only.
No full-budget training or test accuracy is claimed for these new variants.**

## Repository layout

- reproduce/: runnable experiment trainers, reporting, verified data downloader,
  and licensed AugMix operations.
- configs/: explicit experiment settings; the current budget is10 epochs/task.
- tests/: objective,decoder,checkpoint and historical-provenance tests.
- scripts/: configuration launcher and completed-result summary.
- docs/: current objective specification; history/ preserves previous lab notes.
- results/:11 completed historical evaluations,selected provenance,and exact
  hashed source snapshots. No datasets or model weights are committed.

Historical runners are preserved separately to avoid silently changing established
experiments. The new work is in [augce_cifar10.py](reproduce/augce_cifar10.py).

## Setup

Tested with Python3.13.11,PyTorch2.10.0+cu128,torchvision0.25.0+cu128,
NumPy2.4.2,Pillow12.1.1,Matplotlib3.10.8 on an RTX3060Ti under Ubuntu/WSL.
Python3.12+ is recommended. Install a PyTorch build suitable for your device;
the commands below show the tested CUDA12.8 build:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
    python -m pip install -r requirements.txt

Use PowerShell's .venv/Scripts/Activate.ps1 for a native Windows environment.
CUDA,MPS,and CPU are explicit device choices in the new runner; unavailable
accelerators cause an error. Only CUDA was tested for the new decoder path here.
The historical MetaMix runner requires CUDA.

## Data

Run from the repository root:

    python reproduce/download_cifar10_benchmark.py

The downloader verifies clean CIFAR10 and all 15 standard CIFAR10-C arrays,
including the corruption archive MD5 and individual array SHA256 hashes.
The default directory is experiments/metamix_cifar10/data.
To reuse downloaded data elsewhere, pass --data-root to the configuration launcher.
No data is committed to this repository.

## New experiment: augmented CE, with optional reconstruction

Both variants keep:
- clean current-batch CE;
- DER++ replay CE(weight.5) and stored-logit mean MSE(weight.2);
- the same random AugMix views and independent RNG streams;
- the same joint original/view1/view2 BatchNorm forward and separate replay-logit forward.

Class-aware InfoNCE is removed from these variants. Instead, add the mean CE
over the two augmented views of current+CE-replay examples (coefficient1).

The optional decoder reconstructs **each view's own input** from its raw pooled
128-dimensional representation: clean->clean,augmented->augmented.
Pixel reconstruction uses ordinary mean MSE in[0,1],with coefficient1 as an
unvalidated starting choice. Gradients update both encoder and decoder.
The decoder is unused during classification evaluation.

Print a command without running training:

    python scripts/run_experiment.py configs/augce.json
    python scripts/run_experiment.py configs/augce_decoder.json

Launch only when desired:

    python scripts/run_experiment.py configs/augce.json --run
    python scripts/run_experiment.py configs/augce_decoder.json --run

Reuse an existing dataset:

    python scripts/run_experiment.py configs/augce_decoder.json --data-root /path/to/data --run

Use distinct output directories when changing any setting. Checkpoint/source
compatibility guards intentionally reject mismatches. Configuration files are
starting recipes,not tuned optima. See [the exact objective](docs/augmented_ce_decoder.md).

## Training and evaluation

Default:seed17,all 50,000 CIFAR10 training images,5 sequential class pairs,
10 epochs/task (50 total),current batch32 including short batches,reservoir500,
constant SGD lr.03 without momentum/weight decay. This is15,650 updates and
500,000 current presentations. Training uses clean training data and its random views.

Final evaluation is on all 10,000 clean test examples and 750,000 CIFAR10-C
corrupted examples (15 corruptions x5 severities x10,000).
Class-IL chooses among all 10 classes; task-IL receives the true class-pair mask.
No augmented views or decoder are used at test. No test-based checkpoint selection.

Outputs:config.json,train_indices.json,source snapshot,trace.jsonl,
progress.json,diagnostics.png,checkpoint.pt,evaluation.json,complete.json.
Checkpoints include optimizer,CPU reservoir,all RNG streams,and epoch position.
Only complete epochs resume;uncommitted trace rows are removed.

## Validation

    python -m unittest discover -s tests -v

These are short synthetic correctness tests,including tiny5-task pipeline runs.
They do not download data or launch50-epoch training.
Tests use CUDA if available,otherwise CPU. This is a test-device choice;the actual
trainer never silently changes the explicitly requested device.

## Completed experiments

    python scripts/summarize_results.py

The full results and three-seed feature-MSE ablation are in [results](results/README.md).
The strongest completed seed17 class-IL result was different-class-negative InfoNCE:
51.59% clean and 45.27% CIFAR10-C. This is a single-seed exploratory result.
MetaMix uses a paper-guided first-order implementation,not verified author code.
A full matched AugMix+JS baseline has not been run.

## Attribution

The included Google AugMix PIL operations retain their Apache2.0 license.
See [third-party notices](THIRD_PARTY_NOTICES.md) and [license](reproduce/AUGMIX_LICENSE).
CIFAR10/CIFAR10-C are separate datasets and not distributed here.
