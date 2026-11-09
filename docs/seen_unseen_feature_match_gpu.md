# Seen-to-unseen feature matching: GPU handoff

Run commands from the root of **robust-continual-learning**, not the separate
ce7554-ai-image-classification repository. This is a new experiment; use a fresh
output directory. Historical JS/InfoNCE modes retain their original objectives.

## Train SmallCNN with both mixer terms

After the environment and dataset setup below, run:

```sh
python scripts/run_experiment.py configs/seen_js_feature_match.json --run
```

This recipe minimizes the sum of two equally weighted losses. JS compares the
classifier predictions on the original image and both mixer-produced seen views.
Feature MSE compares the seen mixtures to detached features of unmixed unseen
chains. The classifier stays fixed during both mixer terms. The trace records
the sum in `mixer_loss_after` and its MSE component in `feature_match_after`.
Results go to `experiments/seen_js_feature_match_smallcnn/seed17`.

## Feature matching alone: objective and update order

Use the chain-aware controller with fixed original-image weight m=0.5 and three
learned chain weights per image and view. Each batch splits the nine AugMix
operations into disjoint seen (5–7) and unseen (2–4) pools, resampled next batch.
Both pools operate on current training images plus replay-CE images, never test data.

1. Sample two views of three seen chains and two views of three unseen chains.
2. Train the classifier using original-image current CE + 0.5 replay CE + 0.2
   stored-logit MSE + seen-view consistency. The supplied configuration uses
   class-aware InfoNCE at temperature 0.1; JS is also supported.
3. Fix the updated classifier in evaluation mode. Recompute detached descriptors
   of the **seen** chains for the controller. Extract and detach features of all
   six **unmixed unseen chains**, without passing them through the mixer.
4. Perform three mixer SGD updates at learning rate 0.05. Reuse sampled chains,
   target features, clean context and incoming recurrent state across these steps.
   Each update recomputes seen-chain weights and mixed-image features.

For seen mixed image s[v,i] = 0.5*x[i] + 0.5*sum_k w[v,i,k]*a_seen[v,k](x[i]),
the outer loss is exactly:

    L_match = mean_(v,k,i,d) (h_theta(s[v,i])[d] - stopgrad(h_theta(u[v,k,i]))[d])**2

Here v=1,2 and k=1,2,3. h is the classifier's raw penultimate pooled feature,
128 dimensions for SmallCNN or 512 for ResNet18. The coefficient is 1. Features
are not normalized. Gradients flow through the fixed classifier into the seen
mixing weights only. There is no derivative through the classifier optimizer step.
There is no unseen JS/InfoNCE loss in this mode and no outer CE term.
Unseen views are used only as detached targets: they bypass the mixer and do not
contribute gradients or BatchNorm updates to the classifier. Seen-view InfoNCE
(or JS when selected) remains active in the classifier loss.

The saved configuration records `classifier_seen_consistency`, its coefficient,
`feature_match_coefficient`, `unseen_targets_use_mixer=false`, and
`unseen_targets_update_classifier=false`. The obsolete
`old_unseen_consistency_coefficient` field has been removed to avoid implying
that an optional unseen JS computation exists in this mode.

The classifier continues to train on seen mixtures. Since its update precedes the
mixer update, new mixer parameters affect subsequent batches. Persistent recurrent
state advances using the pre-mixer-update transition, as in existing experiments.

The mean loss matches every seen view to three targets, which can encourage an
average representation when targets conflict. Fixed m and the convex seen-chain
mixture limit what can be matched. This is a starting experiment, not a claim of
pixel similarity or improved robustness.

## Get the repository

On the GPU computer, clone the repository and enter its root:

```sh
git clone https://github.com/Riezmann75/robust-continual-learning.git
cd robust-continual-learning
```

For an existing clean checkout on main, update it instead:

```sh
git pull --ff-only origin main
```

The new loss, SmallCNN configuration and tests are included in the repository;
no separate patch is needed.

## Environment (NVIDIA CUDA)

Use Python 3.12 or 3.13 and an NVIDIA driver compatible with the CUDA wheel.
The following matches the repository's pinned PyTorch 2.10 / torchvision 0.25
CUDA 12.8 setup. On Linux:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
```

On Windows PowerShell, create the environment with `py -3.12 -m venv .venv` and
activate it with `.\.venv\Scripts\Activate.ps1`; the subsequent Python commands
are the same. Training explicitly requires CUDA and does not fall back to CPU.

## Verify before a full run

```sh
python -m unittest discover -s tests -p test_seen_unseen_feature_match.py -v
python reproduce/chain_aware_mixing.py --pipeline-test --outer-consistency feature_match --consistency infonce --backbone smallcnn --output experiments/feature_match_smallcnn_smoke/seed17
```

The synthetic smoke test runs five tiny tasks and evaluation without a dataset
download. It checks execution, not accuracy. Use a fresh smoke output path after
changing code. The unit tests check the formula, detached targets, frozen
classifier, nonzero mixer gradients, seen-only outer mixing, and replay/short batches.

## Download and train

```sh
python reproduce/download_cifar10_benchmark.py
python scripts/run_experiment.py configs/seen_unseen_feature_match.json
python scripts/run_experiment.py configs/seen_unseen_feature_match.json --run
```

The second command prints the full command without running it. The supplied
configuration trains the small six-convolution CNN **from scratch** (288,746
parameters, 128-dimensional features), seed 17, ten epochs
per task over five tasks. It uses all 50,000 training images, batch 32, replay 32,
a 500-image reservoir, classifier SGD 0.03 and mixer SGD 0.05. Training ends with
clean and CIFAR-10-C evaluation. Test data do not select checkpoints.

To reuse verified data:

```sh
python scripts/run_experiment.py configs/seen_unseen_feature_match.json --data-root /path/to/data --run
```

For a single GPU selection, Linux supports `CUDA_VISIBLE_DEVICES=0` before the
training command. In PowerShell set `$env:CUDA_VISIBLE_DEVICES="0"` first.

## Outputs and resume

The default full-run directory is
`experiments/seen_unseen_feature_match_smallcnn/seed17`.
Inspect `trace.jsonl`, `progress.json`, `weights_progress.png`, and finally
`evaluation.json` and `complete.json`. New metrics include `feature_match_after`
and each mixer update's `feature_match`; the generic outer-consistency fields
also contain feature MSE when `outer_consistency` is `feature_match`.

Rerun the identical training command to resume from the latest completed epoch.
Partial-epoch trace rows are discarded. The checkpoint is
`MetaMix/checkpoint.pt` under the run directory. Keep source, configuration,
dataset path, software and GPU unchanged for exact configuration matching;
changing these may be rejected. Do not resume a historical JS/InfoNCE checkpoint
with the new loss. Use a new output directory when changing loss or backbone.
