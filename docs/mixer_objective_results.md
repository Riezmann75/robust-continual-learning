# Feature MSE as a mixer objective

Feature MSE gives the mixer a direct target for selecting augmentation weights.
For each image, the mixer blends seen chains. The frozen classifier encodes that
mixture and three independently sampled, unmixed pseudo-unseen targets. The mixer
minimizes their mean squared feature distance. Class-aware InfoNCE trains the
classifier on the original and two seen mixtures.

## Batch-32 SmallCNN results

All rows use class-aware InfoNCE for classifier consistency. The feature-matching
row is associated with the batch-32 recipe from the run history. Its evaluation
file contains scores only. Confirmation of the batch size and whether the mixer
also used seen-view JS requires the original run configuration.

| Mixer and objective | Clean Class-IL | CIFAR-10-C Class-IL | Clean Task-IL | CIFAR-10-C Task-IL |
| --- | ---: | ---: | ---: | ---: |
| [Feature matching](../results/feature_matching_evaluation/evaluation.json) | 54.96 | 45.05 | 90.17 | 84.72 |
| [MetaMix-guided, JS](../results/corrected_mixing_seed17/learned_infonce_js/evaluation.json) | 48.26 | 38.55 | 89.25 | 82.69 |
| [MetaMix-guided, InfoNCE](../results/corrected_mixing_seed17/learned_infonce/evaluation.json) | 49.42 | 39.68 | 90.08 | 83.24 |
| [Random mixing](../results/corrected_mixing_seed17/random_infonce/evaluation.json) | 54.30 | 48.27 | 91.20 | 87.19 |

Class-IL accuracy (%) by corruption family, averaged across all five severities.

| Mixer and objective | Noise | Blur | Weather | Digital |
| --- | ---: | ---: | ---: | ---: |
| Feature matching | 42.02 | 46.40 | 44.95 | 46.08 |
| MetaMix-guided, JS | 34.71 | 40.06 | 39.16 | 39.31 |
| MetaMix-guided, InfoNCE | 36.42 | 40.35 | 40.54 | 40.59 |
| Random mixing | 48.78 | 49.96 | 47.48 | 46.98 |

Feature matching reached **45.05% corrupted Class-IL**, above the learned JS
(38.55%) and InfoNCE (39.68%) baselines. Random mixing reached **48.27%**.
On clean Class-IL, feature matching reached **54.96%**, compared with **54.30%**
for random mixing. The learned baselines also differ in their controller inputs
and original-image weights, so these differences describe whole recipes. An
estimate of MSE's contribution requires a comparison with the same controller.
