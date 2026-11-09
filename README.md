# Robust continual learning

Controlled Split-CIFAR10 experiments using a small CNN, DER++ replay, and
clean/augmented-view objectives. Includes completed historical results, corrected
random/learned mixing comparisons, and augmented-CE/reconstruction runners.

The comparison below includes five corrected-controller/control runs and three
chain-aware variants, all completed. Other experiment families
are documented separately; their objectives and results should not be conflated.

## Repository layout

- libs/: shared model architectures (`models.py`), objectives (`loss.py`), and
  batch updates (`trainer.py`: `SingleLevelTrainer` and `BiLevelTrainer`).
- reproduce/: runnable experiment trainers, reporting, verified data downloader,
  and licensed AugMix operations.
- configs/: explicit experiment settings; the current budget is10 epochs/task.
- tests/: objective,decoder,checkpoint and historical-provenance tests.
- scripts/: configuration launcher and completed-result summary.
- docs/: current objective specification; history/ preserves previous lab notes.
- results/: 22 completed evaluations, full configurations, completion records,
  and verified source snapshots. No datasets or model weights are committed.

Historical runners are preserved separately to avoid silently changing established
experiments. The new work is in [augce_cifar10.py](reproduce/augce_cifar10.py).

### Shared training library

The runners keep their command-line options, data preparation, augmentation RNGs,
replay buffers, reporting, and checkpoint formats. They import model and loss
implementations from `libs` and use its trainers for batch updates. Existing
runner imports such as `Network`, `Controller`, and `step` remain supported.
Different objective masks and controller architectures retain separate definitions.

`SingleLevelTrainer` holds a classifier, optimizer, replay buffer, and random
streams. `BiLevelTrainer` also holds the controller, its optimizer, and recurrent
state. Both expose `step(raw, labels, **options)`; the bi-level trainer updates
its `state` after a successful batch and returns the metrics (plus debug data
when requested). For example, with resources initialized as in an existing runner:

```python
from libs.trainer import SingleLevelTrainer, BiLevelTrainer

trainer = SingleLevelTrainer(model, optimizer, memory, streams, device,
                             update=augce.step, lambda_aug=1., lambda_recon=0.)
metrics = trainer.step(raw, labels)

trainer = BiLevelTrainer(model, controller, optimizer, mixer_optimizer,
                         memory, state, device, update=matched_mixing.step,
                         mixer="learned", kind="js")
metrics = trainer.step(raw, labels)
state = trainer.state  # Save with the existing checkpoint payload.
```

The `update_*` methods in `libs/trainer.py` contain the experiment-specific update
implementations. Data, augmentation, and objective functions are explicit hooks,
supplied by the compatibility `step` functions. New source snapshots include
`libs/` and its hashes; archived result snapshots remain unchanged. Existing run
directories still enforce exact source/configuration matching, so use a new output
directory for runs made with refactored code.

## One-step bilevel SmallCNN mixer

Run `python scripts/run_experiment.py configs/virtual_mixer_smallcnn.json --run`.
The [runner](reproduce/virtual_mixer.py) differentiates through one classifier SGD
update on original and pseudo-seen training views (CE, DER++ replay, and
class-aware InfoNCE). The mixer minimizes the sum of clean and pseudo-unseen
cross-entropies on a separate labeled current-task query batch. Each query image
receives one random unseen chain. The classifier commits its support gradient
once. Each batch has one mixer update, and the original-image weight stays at 0.5.

The seed-17 split reserves 500 of each class's 5,000 training images for queries.
This gives the classifier 45,000 distinct support images over five tasks, compared
with 50,000 in earlier runs. The run records both index sets and the query losses.
Classifier steps use up to 256 current examples, 32 replay examples for CE, and an independent replay sample for distillation. Mixer query batches contain 32 examples. Results go to `experiments/virtual_mixer_smallcnn_bs256/seed17`.

## Corrected mixing comparison (seed 17)

For a restatement of our implementation, see [MetaMix paper-to-code guide](docs/metamix_implementation.md): controller inputs, training order, paper-to-code mapping, and remaining differences.

**MetaMix-guided** denotes our paper-guided implementation, not verified author code.

The eight previously documented runs completed **10 epochs per task across five tasks**, using all
50,000 clean training images, the same 288,746-parameter CNN, and CE + DER++.
Each run performed 15,650 classifier updates and 500,000 current-image presentations.

| Mixer | Classifier consistency | Mixer objective | Clean class-IL | CIFAR-10-C class-IL | Clean task-IL | CIFAR-10-C task-IL |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| MetaMix-guided mixer | JS | JS | 46.22% | 37.58% | 89.17% | 83.33% |
| Random mixer | JS | None | 48.80% | 42.11% | 89.71% | 85.41% |
| Random mixer | Class-aware InfoNCE | None | 54.30% | **48.27%** | **91.20%** | **87.19%** |
| MetaMix-guided mixer | Class-aware InfoNCE | Class-aware InfoNCE | 49.42% | 39.68% | 90.08% | 83.24% |
| MetaMix-guided mixer | Class-aware InfoNCE | JS | 48.26% | 38.55% | 89.25% | 82.69% |
| Chain-aware mixer (`m = 0.5`) | Class-aware InfoNCE | JS | **56.22%** | 47.17% | 91.02% | 86.24% |
| Chain-aware mixer (`m = 0.5`) | JS | JS | 50.33% | 42.37% | 90.50% | 85.66% |
| Chain-aware mixer (`m = 0.5`) | Class-aware InfoNCE | Class-aware InfoNCE | 50.15% | 42.79% | 90.99% | 86.01% |
| Feature-matching run* | Class-aware InfoNCE | Feature MSE (JS unconfirmed) | 54.96% | 45.05% | 90.17% | 84.72% |

*The imported feature-matching evaluation scored 1.26 percentage points lower on clean Class-IL and 2.12 points lower on corrupted Class-IL than the chain-aware InfoNCE / JS baseline. Both feature-matching recipes use classifier InfoNCE. The supplied scores do not identify whether the mixer also used seen-view JS. [Raw evaluation](results/feature_matching_evaluation/evaluation.json).

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
negatives; other same-class examples are excluded. Classifier consistency uses
pseudo-seen views; mixer objectives use pseudo-unseen views. The unchanged-controller InfoNCE/JS row changes
only the outer consistency objective from InfoNCE to JS relative to learned InfoNCE.
The corrected LSTM controller is unchanged: m and w remain learned, with no chain inputs.
Random mixers have no outer optimization. All consistency coefficients are 1;
InfoNCE temperature is 0.1. See [implementation details](docs/matched_mixing.md).

Random InfoNCE exceeds learned InfoNCE by **8.59 percentage points** on corrupted
class-IL. Final-epoch mean original-image weight m was **0.9726** for learned JS
and **0.9916** for learned InfoNCE, indicating weak augmentation contributions.
These are observations from one seed, not proof that the original authors' method
collapses. Our CNN, 10-epoch budget and BatchNorm convention differ from the paper;
the first-order mixer-gradient interpretation remains unverified against author code.

The new InfoNCE/JS run scores **1.16 points lower on clean class-IL** and
**1.13 points lower on corrupted class-IL** than InfoNCE/InfoNCE. Its final-epoch
mean m is **0.9739**, with **99.26%** of per-example values above 0.95. Changing
only the outer objective did not improve this seed or prevent weak augmentation.
See [the objective-isolation experiment](docs/matched_infonce_js.md).

Sampling fingerprints match at every update across four runs
(random JS, random InfoNCE, learned InfoNCE, learned InfoNCE/JS). The learned-JS reference predates
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
python scripts/run_experiment.py configs/matched_infonce_js.json --data-root /path/to/data --run
```

The chain-aware variants additionally change controller inputs, fix m=0.5, and
assign separate weights per view. Comparing them with the earlier mixer is not
an isolated outer-objective ablation. See [the implementation](docs/chain_aware_mixing.md).

The JS classifier comparison keeps the chain-aware controller and outer JS fixed:
`python scripts/run_experiment.py configs/chain_aware_js.json --run`.
It writes to `experiments/chain_aware_js/seed17`. The completed JS / JS run reaches
50.33% clean and 42.37% corrupted class-IL, respectively 5.89 and 4.80 percentage
points below the chain-aware InfoNCE / JS run. These are single-seed results.

The InfoNCE / InfoNCE comparison changes only the mixer objective relative to
chain-aware InfoNCE / JS. It reaches 50.15% clean and 42.79% corrupted class-IL,
respectively 6.07 and 4.38 percentage points lower. Reproduce it with
`python scripts/run_experiment.py configs/chain_aware_infonce_infonce.json --run`.

### ResNet18 results (seed 17, trained from scratch)

The following eight runs use CIFAR-adapted ResNet18 with **no pretrained weights**,
native 512-dimensional features, and 11,173,962 classifier parameters. Each run
completed **10 epochs per task across five tasks**, 15,650 classifier updates,
and 500,000 current-image presentations, retaining the CE + DER++ settings above.
These are single-seed results with inherited settings, not ResNet18-tuned results
or a reproduction of the paper's 50-epochs-per-task budget.

All values are **accuracy (%)** from the final checkpoint, evaluated on 10,000
clean test images and 750,000 CIFAR-10-C images (15 corruptions x 5 severities).
Class-IL predicts among all ten classes; Task-IL receives the correct class pair.
**Bold** marks the best score in each reported accuracy column.

| Mixer | Classifier consistency | Mixer objective | Clean class-IL | CIFAR-10-C class-IL | CIFAR-10-C task-IL |
| --- | --- | --- | ---: | ---: | ---: |
| MetaMix-guided mixer | JS | JS | 25.79% | 24.15% | 76.86% |
| Random mixer | JS | None | 33.46% | 31.29% | 83.57% |
| Random mixer | Class-aware InfoNCE | None | **41.73%** | **38.96%** | **84.30%** |
| MetaMix-guided mixer | Class-aware InfoNCE | Class-aware InfoNCE | 28.93% | 26.86% | 76.16% |
| MetaMix-guided mixer | Class-aware InfoNCE | JS | 29.28% | 26.90% | 77.73% |
| Chain-aware mixer (`m = 0.5`) | Class-aware InfoNCE | JS | 32.47% | 30.39% | 80.52% |
| Chain-aware mixer (`m = 0.5`) | JS | JS | 28.03% | 26.37% | 79.93% |
| Chain-aware mixer (`m = 0.5`) | Class-aware InfoNCE | Class-aware InfoNCE | 31.40% | 29.41% | 80.65% |

Run provenance: PBS array `26155568[]`, NVIDIA A100-SXM4-40GB, Python 3.12.11,
PyTorch 2.8.0+cu129 and torchvision 0.23.0+cu129. These versions differ from the
pinned setup versions. Downloaded evaluations and logs were checked locally;
the HPC artifacts are not committed to Git. See the
[ResNet18 HPC instructions](jobs/resnet18_readme/README.md) for the eight-setting
mapping and reproduction commands.

### Class-IL and Task-IL by corruption category

[Category and per-corruption tables](docs/corruption_category_results.md) report
both evaluation modes for the eight documented runs and the imported evaluation, grouped as in MetaMix Tables 6-7:
Noise (Gaussian, shot, impulse), Blur (defocus, glass, motion, zoom), Weather
(snow, frost, fog, brightness), and Digital (contrast, elastic, pixelate, JPEG).
Each corruption averages severities 1-5. Overall averages all 15 corruptions,
so the four category means are weighted by their sizes (3,4,4,4).

Our fixed class pairs are (0,1), (2,3), (4,5), (6,7), (8,9). The five two-class
task structure matches the paper, but its exact class order is not verified.
Task-IL is given the correct pair; Class-IL is not. These results come from the
existing full evaluations, not retraining or paper-equivalent accuracy claims.

```sh
python scripts/summarize_corruption_categories.py
```

<!-- core-corruption-ranked-tables -->
All values are **accuracy (%)**, averaged over severities 1-5. **Bold** marks the best and <ins>underline</ins> marks the second-best score in each column, ranked using unrounded values. Objectives are **classifier consistency / mixer objective**; all methods include CE + DER++.

#### Class-IL — all ten classes

| Mixer | Objectives | Noise | Blur | Weather | Digital | Overall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MetaMix-guided mixer | JS / JS | 32.75 | 39.33 | 38.63 | 38.40 | 37.58 |
| Random mixer | JS / none | 40.38 | 44.00 | 41.69 | 41.95 | 42.11 |
| Random mixer | InfoNCE / none | **48.78** | **49.96** | <ins>47.48</ins> | <ins>46.98</ins> | **48.27** |
| MetaMix-guided mixer | InfoNCE / InfoNCE | 36.42 | 40.35 | 40.54 | 40.59 | 39.68 |
| MetaMix-guided mixer | InfoNCE / JS | 34.71 | 40.06 | 39.16 | 39.31 | 38.55 |
| Chain-aware mixer (`m = 0.5`) | InfoNCE / JS | <ins>45.06</ins> | <ins>47.26</ins> | **47.51** | **48.31** | <ins>47.17</ins> |
| Chain-aware mixer (`m = 0.5`) | JS / JS | 36.79 | 45.67 | 42.39 | 43.25 | 42.37 |
| Chain-aware mixer (`m = 0.5`) | InfoNCE / InfoNCE | 39.84 | 44.79 | 42.53 | 43.26 | 42.79 |
| Feature-matching run* | InfoNCE / Feature MSE (JS unconfirmed) | 42.02 | 46.40 | 44.95 | 46.08 | 45.05 |

#### Task-IL — correct class pair supplied

| Mixer | Objectives | Noise | Blur | Weather | Digital | Overall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MetaMix-guided mixer | JS / JS | 81.11 | 82.76 | 85.06 | 83.85 | 83.33 |
| Random mixer | JS / none | 83.22 | 85.60 | 86.44 | 85.84 | 85.41 |
| Random mixer | InfoNCE / none | **86.23** | **87.03** | **88.23** | **87.03** | **87.19** |
| MetaMix-guided mixer | InfoNCE / InfoNCE | 81.53 | 82.69 | 84.75 | 83.57 | 83.24 |
| MetaMix-guided mixer | InfoNCE / JS | 79.01 | 82.57 | 84.74 | 83.52 | 82.69 |
| Chain-aware mixer (`m = 0.5`) | InfoNCE / JS | <ins>85.79</ins> | 85.46 | <ins>87.50</ins> | 86.11 | <ins>86.24</ins> |
| Chain-aware mixer (`m = 0.5`) | JS / JS | 82.96 | <ins>85.78</ins> | 86.92 | <ins>86.30</ins> | 85.66 |
| Chain-aware mixer (`m = 0.5`) | InfoNCE / InfoNCE | 84.69 | 85.54 | 87.42 | 86.04 | 86.01 |
| Feature-matching run* | InfoNCE / Feature MSE (JS unconfirmed) | 83.49 | 84.04 | 86.06 | 84.98 | 84.72 |

<!-- /core-corruption-ranked-tables -->

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

The README focuses on the six core variants above. For the broader history, see
[all 22 completed runs](docs/all_training_results.md) and
[their Class-IL/Task-IL corruption tables](docs/all_settings_corruptions.md).

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

## ResNet18 from scratch

All runners supported by `scripts/run_experiment.py` accept
`--backbone resnet18`; `smallcnn` remains the default. The historical
`metamix_long.py` and `metamix_revised.py` entry points retain SmallCNN.
ResNet18 is created with `torchvision.models.resnet18(weights=None)` and never
loads pretrained weights. For 32x32 CIFAR inputs it uses a 3x3, stride-1 stem
and no initial maxpool. Native 512-dimensional pooled features feed the classifier and feature losses;
the decoder and controller input dimensions adapt to the selected backbone. This is a CIFAR-adapted
ResNet18, with a ten-class head.

After the dataset setup above, run:

```sh
python scripts/run_experiment.py configs/augce_resnet18.json --run
python scripts/run_experiment.py configs/augce_decoder_resnet18.json --run
```

Or select ResNet18 for another supported configuration:

```sh
python scripts/run_experiment.py configs/matched_random_infonce.json \
  --backbone resnet18 --device cuda --data-root /path/to/data \
  --output experiments/random_infonce_resnet18/seed17 --run
```

Use a fresh output directory when changing backbones. Configuration metadata
records the architecture, scratch initialization and actual parameter counts;
resume guards reject incompatible runs. Existing optimizer settings are retained
as starting recipes and have not been tuned for ResNet18. The earlier SmallCNN results and the separate ResNet18 table above report
different backbones and should be interpreted accordingly.

PBS jobs for all eight main-comparison settings with scratch ResNet18, including
post-clone setup and submission commands: [HPC instructions](jobs/resnet18_readme/README.md).
