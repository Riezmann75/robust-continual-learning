# Robust continual learning

Controlled Split-CIFAR10 experiments using a small CNN, DER++ replay, and
clean/augmented-view objectives. Includes completed historical results, corrected
random/learned mixing comparisons, and augmented-CE/reconstruction runners.

The corrected four-run comparison below is complete. Other experiment families
are documented separately; their objectives and results should not be conflated.

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

## Corrected mixing comparison (seed 17)

All four runs completed **10 epochs per task across five tasks**, using all
50,000 clean training images, the same 288,746-parameter CNN, and CE + DER++.
Each run performed 15,650 classifier updates and 500,000 current-image presentations.

| Mixer | Consistency objective | Clean class-IL | CIFAR-10-C class-IL | Clean task-IL | CIFAR-10-C task-IL |
|---|---|---:|---:|---:|---:|
| Learned | JS (paper-guided MetaMix implementation) | 46.22% | 37.58% | 89.17% | 83.33% |
| Random | JS | 48.80% | 42.11% | 89.71% | 85.41% |
| Random | Class-aware InfoNCE | **54.30%** | **48.27%** | **91.20%** | **87.19%** |
| Learned | Class-aware InfoNCE | 49.42% | 39.68% | 90.08% | 83.24% |

Evaluation uses the final checkpoint on **10,000 clean test images** and
**750,000 CIFAR-10-C images** (15 corruptions x 5 severities x 10,000 images).
Class-IL predicts among all ten outputs. Task-IL is supplied the true class pair.
No augmentation is applied at test, and test accuracy does not select checkpoints.

Corrections include augmentation of current and replay examples, individual
controller contexts and recurrent batch slots, pre-solver state advancement,
and a joint clean/two-augmented-view BatchNorm forward. Independent logit replay
still uses a separate forward. Pseudo-seen and pseudo-unseen operation pools are
resampled each batch; both use training images, never CIFAR-10-C test data.

Class-aware InfoNCE uses two same-image positive views and different-class
negatives; other same-class examples are excluded. In these four runs, the named
consistency objective is used for both classifier and learned-mixer updates.
Random mixers have no outer optimization. All consistency coefficients are 1;
InfoNCE temperature is 0.1. See [implementation details](docs/matched_mixing.md).

Random InfoNCE exceeds learned InfoNCE by **8.59 percentage points** on corrupted
class-IL. Final-epoch mean original-image weight m was **0.9726** for learned JS
and **0.9916** for learned InfoNCE, indicating weak augmentation contributions.
These are observations from one seed, not proof that the original authors' method
collapses. Our CNN, 10-epoch budget and BatchNorm convention differ from the paper;
the first-order mixer-gradient interpretation remains unverified against author code.

Sampling fingerprints match at every update across the three new runs
(random JS, random InfoNCE, learned InfoNCE). The learned-JS reference predates
fingerprint logging. Random weights are independent per image and view, whereas
learned weights are shared across the two views, so this is not a perfectly
isolated mixer-only ablation.

[Full evaluations, configurations and exact source snapshots](results/corrected_mixing_seed17/)
are included without datasets, checkpoints or full batch traces.

Reproduce each run from the repository root (CUDA required):

```sh
python scripts/run_experiment.py configs/metamix_jointbn.json --data-root /path/to/data --run
python scripts/run_experiment.py configs/matched_random_js.json --data-root /path/to/data --run
python scripts/run_experiment.py configs/matched_random_infonce.json --data-root /path/to/data --run
python scripts/run_experiment.py configs/matched_learned_infonce.json --data-root /path/to/data --run
```

The later chain-aware fixed-m controller is a separate experiment and is not
included in this four-run comparison.

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
The corrected MetaMix/matched-mixing tests require CUDA. Some older tests support
CPU. Actual trainers never silently change the explicitly requested device.

## Completed experiments

    python scripts/summarize_results.py

The full results and three-seed feature-MSE ablation are in [results](results/README.md).
The older different-class-negative InfoNCE result was 51.59% clean and 45.27%
CIFAR10-C. The corrected comparison above uses a different augmentation pipeline
and should be considered separately from those historical runs.
MetaMix uses a paper-guided first-order implementation,not verified author code.
The corrected random-mixer JS comparison is now complete (see above).

## Attribution

The included Google AugMix PIL operations retain their Apache2.0 license.
See [third-party notices](THIRD_PARTY_NOTICES.md) and [license](reproduce/AUGMIX_LICENSE).
CIFAR10/CIFAR10-C are separate datasets and not distributed here.
