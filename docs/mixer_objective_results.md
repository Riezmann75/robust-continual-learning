# Feature MSE as a mixer objective

Feature MSE gives the mixer a direct target for selecting augmentation weights.
For each image, the mixer blends seen chains. The frozen classifier encodes that
mixture and three independently sampled, unmixed pseudo-unseen targets. The mixer
minimizes their mean squared feature distance. Class-aware InfoNCE trains the
classifier on the original and two seen mixtures.

## Batch-256 comparison

The reported MSE recipe scored **6.44 percentage points higher on corrupted
Class-IL** and **8.38 points higher on clean Class-IL** than the one-step bilevel
recipe. The earlier feature-matching score provides batch-size context. Percent
accuracies below come from the linked evaluation files. Recipe labels follow the
run commands associated with the reports. The files contain scores only, and
the earlier evaluation does not distinguish MSE alone from MSE plus seen-view JS.

| Reported SmallCNN recipe | Clean Class-IL | CIFAR-10-C Class-IL | Clean Task-IL | CIFAR-10-C Task-IL |
| --- | ---: | ---: | ---: | ---: |
| [Feature MSE, batch 256](../results/feature_mse_bs256/evaluation.json) | 45.54 | 37.92 | 85.99 | 80.32 |
| [One-step bilevel query CE, batch 256](../results/bilevel_bs256/evaluation.json) | 37.16 | 31.48 | 81.20 | 77.11 |
| [Earlier feature matching, batch 32 recipe](../results/feature_matching_evaluation/evaluation.json) | 54.96 | 45.05 | 90.17 | 84.72 |

Class-IL accuracy by corruption family, averaged across five severity levels:

| Recipe | Noise | Blur | Weather | Digital |
| --- | ---: | ---: | ---: | ---: |
| Feature MSE, batch 256 | 40.73 | 37.12 | 36.16 | 38.36 |
| One-step bilevel query CE, batch 256 | 30.39 | 31.36 | 32.00 | 31.89 |
| Earlier feature matching, batch 32 recipe | 42.02 | 46.40 | 44.95 | 46.08 |

## What the comparison supports

The MSE recipe improved all four corruption-family averages over the batch-256
bilevel recipe. This supports keeping feature MSE as a candidate when training
SmallCNN to handle unseen transformations. The recipes also differ in classifier
training data (50,000 versus 45,000 images), classifier updates (2,000 versus
1,800), and mixer steps per classifier update (three versus one). These runs
therefore measure the combined effect of those choices.

The random mixer with classifier InfoNCE reached **48.27% corrupted Class-IL**
in the [batch-32 baseline comparison](../README.md#corrected-mixing-comparison-seed-17),
above both reported feature-matching scores. To estimate the contribution of the
MSE term itself, compare MSE, JS, and MSE + JS with the same controller, data
split, batch size, update counts, and seeds, alongside random mixing.
