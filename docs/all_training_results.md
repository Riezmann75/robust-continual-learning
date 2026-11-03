# Results for all completed training settings

This index covers **22 completed runs**, including repeated seeds, across all
currently completed training settings. It separates pipeline families: rows in
different families are not controlled single-factor comparisons. All results are
our implementations, not verified reproductions of the MetaMix authors' code.

## Shared budget and evaluation

Every included run uses all 50,000 clean CIFAR-10 training images, native 32x32,
five sequential class pairs (0,1), (2,3), (4,5), (6,7), (8,9), 10 epochs/task,
current batch 32 including the last short batch, and 500 reservoir slots.
Each completed 15,650 updates and 500,000 current-image presentations. Base
training includes clean CE and DER++ (replay CE 0.5, stored-logit mean MSE 0.2),
with constant classifier SGD 0.03, no momentum or weight decay.

Evaluate the final checkpoint on 10,000 clean images plus 750,000 CIFAR-10-C
images (15 corruptions, five severities). Class-IL predicts among all ten classes;
Task-IL receives the true class pair. No test-time mixing or test-based selection.
Accuracy columns are percentages. Parenthesized objective weights are coefficients,
not accuracies. Each row links its original configuration, preserving exact settings.

## What differs between families

- Historical MetaMix pilot: batch-pooled controller and earlier replay/BatchNorm
  conventions. It is not the corrected MetaMix reference.
- Random AugMix objectives: all nine operations available without the corrected
  pseudo-seen/unseen split; independent per-image/view random weights and joint
  original/two-view BatchNorm. MSE=0 retains those forwards, so it is not a pure
  no-augmentation DER++ baseline. Normalization/RNG conventions differ from the
  corrected split-pool pipeline.
- Corrected split-pool pipeline: current and replay augmentation, joint forwards,
  and the documented context/state corrections. Learned weights are shared across
  views; random weights are independent across views. Mixer SGD 0.05, J=3 when learned.
- Chain-aware extension: adds candidate descriptors, fixes m=0.5, uses per-view
  learned weights and unseen JS. Multiple components change together.

Most classifiers have 288,746 parameters. The four-head experiment has 305,288
and stores four sets of replay logits per image (more bytes despite 500 slots).
InfoNCE+reconstruction adds a 429,667-parameter decoder during training only.
Feature MSE is dimension-scaled on normalized features; pixel reconstruction uses
ordinary mean MSE. Class-aware InfoNCE excludes other same-class entries from
negatives; standard InfoNCE does not. Both use temperature 0.1. Barlow Twins uses
coordinate scaling /128 and off-diagonal coefficient 0.0051; it is not InfoNCE.

Only feature-MSE and its zero-weight control have three seeds (17,18,19).
Other settings have one seed. Do not infer statistical superiority from these rows.
[Three-seed paired results](../results/three_seed_comparison.json) are retained.

## Historical MetaMix

| Training setting | Seed | Clean class-IL | CIFAR-10-C class-IL | Clean task-IL | CIFAR-10-C task-IL |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Original first-order MetaMix pilot](../results/metamix_seed17/config.json) | 17 | 29.92 | 25.02 | 89.29 | 84.21 |

## Random AugMix objectives

| Training setting | Seed | Clean class-IL | CIFAR-10-C class-IL | Clean task-IL | CIFAR-10-C task-IL |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Feature MSE (1.0)](../results/feature_mse_seed17/config.json) | 17 | 45.29 | 39.88 | 88.49 | 83.76 |
| [Feature MSE (0.0)](../results/feature_mse_seed17_lambda0/config.json) | 17 | 38.27 | 34.59 | 88.66 | 82.34 |
| [Feature MSE (1.0)](../results/feature_mse_seed18/config.json) | 18 | 48.04 | 42.00 | 90.18 | 85.75 |
| [Feature MSE (0.0)](../results/feature_mse_seed18_lambda0/config.json) | 18 | 49.30 | 41.09 | 88.98 | 83.02 |
| [Feature MSE (1.0)](../results/feature_mse_seed19/config.json) | 19 | 47.05 | 41.09 | 89.80 | 84.17 |
| [Feature MSE (0.0)](../results/feature_mse_seed19_lambda0/config.json) | 19 | 43.00 | 36.97 | 89.95 | 82.72 |
| [Standard InfoNCE (1)](../results/infonce_seed17/config.json) | 17 | 48.14 | 42.93 | 88.86 | 85.31 |
| [Class-aware InfoNCE (1)](../results/classaware_infonce_seed17/config.json) | 17 | 51.59 | 45.27 | 91.11 | 86.85 |
| [Class-aware InfoNCE (0.5) + feature MSE (0.5)](../results/mse05_nce05_seed17/config.json) | 17 | 46.00 | 40.97 | 90.46 | 85.57 |
| [Class-aware InfoNCE (1) + feature MSE (0.2)](../results/mse02_nce1_seed17/config.json) | 17 | 47.45 | 42.01 | 90.30 | 86.35 |
| [Augmented CE (1)](../results/augce_seed17/config.json) | 17 | 48.86 | 42.52 | 89.40 | 85.51 |
| [Augmented CE (1) + BT augmented pair (1)](../results/barlow_seed17/config.json) | 17 | 48.12 | 41.10 | 89.33 | 84.74 |
| [BT three pairs (1), no augmented CE](../results/bt_threepair_seed17/config.json) | 17 | 45.25 | 40.59 | 89.32 | 83.94 |
| [Four heads + augmented CE (1) + diversity (0.1)](../results/multihead_seed17/config.json) | 17 | 40.81 | 36.42 | 86.61 | 83.65 |
| [Class-aware InfoNCE (1) + own-view pixel MSE (1)](../results/infonce_recon_seed17/config.json) | 17 | 46.84 | 41.39 | 90.45 | 86.56 |

## Corrected split-pool pipeline

| Training setting | Seed | Clean class-IL | CIFAR-10-C class-IL | Clean task-IL | CIFAR-10-C task-IL |
| --- | ---: | ---: | ---: | ---: | ---: |
| [MetaMix-guided mixer: classifier JS / outer JS](../results/corrected_mixing_seed17/learned_js/config.json) | 17 | 46.22 | 37.58 | 89.17 | 83.33 |
| [Random mixer: classifier JS / no outer loss](../results/corrected_mixing_seed17/random_js/config.json) | 17 | 48.80 | 42.11 | 89.71 | 85.41 |
| [Random mixer: classifier class-aware InfoNCE / no outer loss](../results/corrected_mixing_seed17/random_infonce/config.json) | 17 | 54.30 | 48.27 | 91.20 | 87.19 |
| [MetaMix-guided mixer: classifier class-aware InfoNCE / outer InfoNCE](../results/corrected_mixing_seed17/learned_infonce/config.json) | 17 | 49.42 | 39.68 | 90.08 | 83.24 |
| [MetaMix-guided mixer: classifier class-aware InfoNCE / outer JS](../results/corrected_mixing_seed17/learned_infonce_js/config.json) | 17 | 48.26 | 38.55 | 89.25 | 82.69 |

## Chain-aware extension

| Training setting | Seed | Clean class-IL | CIFAR-10-C class-IL | Clean task-IL | CIFAR-10-C task-IL |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Chain-aware mixer (m=0.5): classifier class-aware InfoNCE / outer JS](../results/corrected_mixing_seed17/chain_aware_fixed_m/config.json) | 17 | 56.22 | 47.17 | 91.02 | 86.24 |

## Settings without a completed full evaluation

The original 100-epochs/task MetaMix run and the intermediate separate-forward
corrected run were stopped; neither has a final accuracy to report. The prepared
augmented-CE plus reconstruction-decoder recipe has no completed full-budget run
in this inventory. It must not be confused with the completed InfoNCE plus
reconstruction setting. Tiny pipeline/smoke tests are excluded.

## Corruption categories and regeneration

[All-settings category and per-corruption tables](all_settings_corruptions.md)
cover both Class-IL and Task-IL for every row above. Original evaluation files
retain every corruption/severity combination; no retraining was required.

```sh
python scripts/summarize_all_settings.py
```

[Machine-readable summary](../results/all_settings_summary.json) and
[artifact manifest](../results/all_settings_manifest.json).
