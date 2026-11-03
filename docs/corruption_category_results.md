# CIFAR-10-C results by category: Class-IL and Task-IL

These tables aggregate the saved full evaluations of seven completed seed-17 runs;
no model was retrained and no new inference or substitute corruptions were needed.
Method labels show **classifier consistency / mixer objective**; all include CE + DER++.
MetaMix-guided denotes our paper-guided implementation, not verified author code.
The chain-aware model also fixes m=0.5 and changes controller inputs and per-view
weighting, so it is not an isolated outer-objective ablation.
The two chain-aware runs differ only in classifier consistency (InfoNCE versus JS);
their controller, fixed m, and outer JS objective remain the same.

## Splits and interpretation

- All configurations use five sequential tasks: (0,1), (2,3), (4,5), (6,7), (8,9).
  These are airplane/automobile, bird/cat, deer/dog, frog/horse, ship/truck in CIFAR-10.
- Each run uses all 50,000 clean training images, the custom CNN, and 10 epochs/task.
- **Class-IL:** predict among all ten output classes without a task identity.
- **Task-IL:** mask outputs to the true class pair. Correct task information is supplied at evaluation; it is not inferred by the model.
- Evaluate the final checkpoint on 10,000 clean and 750,000 corrupted test images;
  no test-time mixing and no test-based checkpoint selection.

The grouping matches [MetaMix supplement, Tables 6 and 7](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Wang_MetaMix_Towards_Corruption-Robust_Continual_Learning_With_Temporally_Self-Adaptive_Data_Transformation_CVPR_2023_supplemental.pdf).
The [main paper's evaluation section](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_MetaMix_Towards_Corruption-Robust_Continual_Learning_With_Temporally_Self-Adaptive_Data_Transformation_CVPR_2023_paper.pdf)
describes five two-class tasks and averaging over five severities. The text checked
does not establish the exact class order used in their runs. Our fixed pairing is
explicit above; identical ordering with the authors is not claimed. Our CNN,
10-epoch budget, single seed and implementation conventions differ from the paper,
so these are our results, not reproductions of its numerical tables.

## Aggregation

For each corruption, average accuracy over severities 1-5 (50,000 predictions).
Each category is the mean of its corruption accuracies. The overall corrupted
average is the mean of all 15 corruptions, **not** an unweighted average of four
categories: Noise has 3 corruptions, while Blur, Weather and Digital each have 4.
Thus overall = (3*Noise + 4*Blur + 4*Weather + 4*Digital)/15.

CIFAR-10 has 1,000 test examples per class, so each two-class task has 2,000.
Our full-test accuracy therefore equals the equally weighted average of the five
task accuracies. The artifacts do not contain a separate per-task breakdown or
backward-transfer metric. Category grouping is an analysis of test corruptions,
not the pseudo-seen/unseen augmentation split used during training.

All values below are **percent accuracy**; calculations use unrounded values.

## Task-IL category summary

| Method | Clean | Noise | Blur | Weather | Digital | Overall corrupted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MetaMix-guided mixer: JS / JS | 89.17 | 81.11 | 82.76 | 85.06 | 83.85 | 83.33 |
| Random mixer: JS / none | 89.71 | 83.22 | 85.60 | 86.44 | 85.84 | 85.41 |
| Random mixer: InfoNCE / none | 91.20 | 86.23 | 87.03 | 88.23 | 87.03 | 87.19 |
| MetaMix-guided mixer: InfoNCE / InfoNCE | 90.08 | 81.53 | 82.69 | 84.75 | 83.57 | 83.24 |
| MetaMix-guided mixer: InfoNCE / JS | 89.25 | 79.01 | 82.57 | 84.74 | 83.52 | 82.69 |
| Chain-aware mixer (m=0.5): InfoNCE / JS | 91.02 | 85.79 | 85.46 | 87.50 | 86.11 | 86.24 |
| Chain-aware mixer (m=0.5): JS / JS | 90.50 | 82.96 | 85.78 | 86.92 | 86.30 | 85.66 |

### Task-IL per-corruption results

#### Noise

| Method | Gauss | Shot | Impulse |
| --- | ---: | ---: | ---: |
| MetaMix-guided mixer: JS / JS | 81.52 | 83.01 | 78.81 |
| Random mixer: JS / none | 83.13 | 84.93 | 81.60 |
| Random mixer: InfoNCE / none | 86.16 | 87.54 | 85.00 |
| MetaMix-guided mixer: InfoNCE / InfoNCE | 81.41 | 83.45 | 79.73 |
| MetaMix-guided mixer: InfoNCE / JS | 78.43 | 81.49 | 77.11 |
| Chain-aware mixer (m=0.5): InfoNCE / JS | 85.58 | 87.19 | 84.60 |
| Chain-aware mixer (m=0.5): JS / JS | 83.11 | 84.97 | 80.79 |

#### Blur

| Method | Defocus | Glass | Motion | Zoom |
| --- | ---: | ---: | ---: | ---: |
| MetaMix-guided mixer: JS / JS | 85.67 | 77.14 | 83.40 | 84.82 |
| Random mixer: JS / none | 88.88 | 79.21 | 86.49 | 87.83 |
| Random mixer: InfoNCE / none | 89.74 | 82.44 | 87.04 | 88.90 |
| MetaMix-guided mixer: InfoNCE / InfoNCE | 86.75 | 76.52 | 82.54 | 84.96 |
| MetaMix-guided mixer: InfoNCE / JS | 85.85 | 77.53 | 82.29 | 84.61 |
| Chain-aware mixer (m=0.5): InfoNCE / JS | 88.35 | 80.99 | 85.39 | 87.10 |
| Chain-aware mixer (m=0.5): JS / JS | 88.88 | 79.55 | 86.79 | 87.89 |

#### Weather

| Method | Snow | Frost | Fog | Bright |
| --- | ---: | ---: | ---: | ---: |
| MetaMix-guided mixer: JS / JS | 83.73 | 83.62 | 84.82 | 88.09 |
| Random mixer: JS / none | 85.08 | 85.03 | 86.48 | 89.18 |
| Random mixer: InfoNCE / none | 87.39 | 87.00 | 87.96 | 90.58 |
| MetaMix-guided mixer: InfoNCE / InfoNCE | 84.66 | 82.86 | 83.22 | 88.26 |
| MetaMix-guided mixer: InfoNCE / JS | 83.21 | 82.99 | 84.71 | 88.06 |
| Chain-aware mixer (m=0.5): InfoNCE / JS | 86.67 | 86.35 | 86.87 | 90.10 |
| Chain-aware mixer (m=0.5): JS / JS | 85.32 | 85.75 | 86.98 | 89.63 |

#### Digital

| Method | Contrast | Elastic | Pixel | JPEG |
| --- | ---: | ---: | ---: | ---: |
| MetaMix-guided mixer: JS / JS | 76.38 | 85.66 | 86.27 | 87.08 |
| Random mixer: JS / none | 80.73 | 87.56 | 87.10 | 87.98 |
| Random mixer: InfoNCE / none | 80.94 | 88.42 | 89.19 | 89.59 |
| MetaMix-guided mixer: InfoNCE / InfoNCE | 72.64 | 86.62 | 87.01 | 88.02 |
| MetaMix-guided mixer: InfoNCE / JS | 76.47 | 85.68 | 85.33 | 86.60 |
| Chain-aware mixer (m=0.5): InfoNCE / JS | 79.25 | 87.43 | 88.96 | 88.79 |
| Chain-aware mixer (m=0.5): JS / JS | 81.13 | 87.84 | 87.47 | 88.77 |

## Class-IL category summary

| Method | Clean | Noise | Blur | Weather | Digital | Overall corrupted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MetaMix-guided mixer: JS / JS | 46.22 | 32.75 | 39.33 | 38.63 | 38.40 | 37.58 |
| Random mixer: JS / none | 48.80 | 40.38 | 44.00 | 41.69 | 41.95 | 42.11 |
| Random mixer: InfoNCE / none | 54.30 | 48.78 | 49.96 | 47.48 | 46.98 | 48.27 |
| MetaMix-guided mixer: InfoNCE / InfoNCE | 49.42 | 36.42 | 40.35 | 40.54 | 40.59 | 39.68 |
| MetaMix-guided mixer: InfoNCE / JS | 48.26 | 34.71 | 40.06 | 39.16 | 39.31 | 38.55 |
| Chain-aware mixer (m=0.5): InfoNCE / JS | 56.22 | 45.06 | 47.26 | 47.51 | 48.31 | 47.17 |
| Chain-aware mixer (m=0.5): JS / JS | 50.33 | 36.79 | 45.67 | 42.39 | 43.25 | 42.37 |

### Class-IL per-corruption results

#### Noise

| Method | Gauss | Shot | Impulse |
| --- | ---: | ---: | ---: |
| MetaMix-guided mixer: JS / JS | 34.12 | 35.89 | 28.25 |
| Random mixer: JS / none | 40.77 | 42.54 | 37.83 |
| Random mixer: InfoNCE / none | 48.50 | 50.78 | 47.06 |
| MetaMix-guided mixer: InfoNCE / InfoNCE | 37.25 | 39.50 | 32.51 |
| MetaMix-guided mixer: InfoNCE / JS | 35.03 | 37.73 | 31.38 |
| Chain-aware mixer (m=0.5): InfoNCE / JS | 45.16 | 47.92 | 42.08 |
| Chain-aware mixer (m=0.5): JS / JS | 37.75 | 39.82 | 32.79 |

#### Blur

| Method | Defocus | Glass | Motion | Zoom |
| --- | ---: | ---: | ---: | ---: |
| MetaMix-guided mixer: JS / JS | 43.32 | 35.15 | 36.27 | 42.58 |
| Random mixer: JS / none | 47.97 | 38.16 | 42.82 | 47.05 |
| Random mixer: InfoNCE / none | 53.12 | 44.62 | 49.18 | 52.92 |
| MetaMix-guided mixer: InfoNCE / InfoNCE | 44.93 | 35.86 | 37.25 | 43.34 |
| MetaMix-guided mixer: InfoNCE / JS | 44.12 | 34.97 | 37.67 | 43.48 |
| Chain-aware mixer (m=0.5): InfoNCE / JS | 52.44 | 40.98 | 45.35 | 50.27 |
| Chain-aware mixer (m=0.5): JS / JS | 50.77 | 36.39 | 45.93 | 49.59 |

#### Weather

| Method | Snow | Frost | Fog | Bright |
| --- | ---: | ---: | ---: | ---: |
| MetaMix-guided mixer: JS / JS | 37.64 | 34.71 | 38.53 | 43.62 |
| Random mixer: JS / none | 41.02 | 37.45 | 42.09 | 46.21 |
| Random mixer: InfoNCE / none | 46.84 | 43.52 | 48.42 | 51.13 |
| MetaMix-guided mixer: InfoNCE / InfoNCE | 40.22 | 36.49 | 39.48 | 45.98 |
| MetaMix-guided mixer: InfoNCE / JS | 38.25 | 34.89 | 40.20 | 43.28 |
| Chain-aware mixer (m=0.5): InfoNCE / JS | 47.50 | 42.61 | 48.49 | 51.44 |
| Chain-aware mixer (m=0.5): JS / JS | 39.77 | 37.84 | 44.17 | 47.76 |

#### Digital

| Method | Contrast | Elastic | Pixel | JPEG |
| --- | ---: | ---: | ---: | ---: |
| MetaMix-guided mixer: JS / JS | 31.78 | 42.22 | 36.46 | 43.13 |
| Random mixer: JS / none | 36.30 | 45.94 | 39.71 | 45.83 |
| Random mixer: InfoNCE / none | 39.44 | 50.80 | 46.85 | 50.85 |
| MetaMix-guided mixer: InfoNCE / InfoNCE | 31.83 | 44.81 | 40.23 | 45.50 |
| MetaMix-guided mixer: InfoNCE / JS | 31.98 | 43.82 | 37.23 | 44.22 |
| Chain-aware mixer (m=0.5): InfoNCE / JS | 38.75 | 50.65 | 51.21 | 52.62 |
| Chain-aware mixer (m=0.5): JS / JS | 37.94 | 48.22 | 39.21 | 47.62 |

## Reproduce the report

```sh
python scripts/summarize_corruption_categories.py
```

[Machine-readable category results](../results/corrected_mixing_seed17/corruption_categories.json) retain fractional accuracies. Original evaluation.json files retain all 75 corruption/severity combinations for both evaluation modes.
