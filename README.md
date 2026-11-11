# Robust continual learning on Split-CIFAR-10

Can a mixer learn which augmentation chains to emphasize from **how well the classifier performs after one training step**? We test this on five sequential CIFAR-10 class pairs with DER++ replay. Our chain-aware mixer scores the effects of individual chains. Our MetaMix baseline uses the learner's context to produce mixing weights. Both now receive gradients through the classifier update.

## MetaMix's bilevel objective

At each training step, MetaMix splits the augmentation operations into a pseudo-seen pool $S$ and a disjoint pseudo-unseen pool $U$. Both pools transform the same mini-batch of current and replay images. For an image $x$, the LSTM mixer with parameters $\phi$ uses learner context to produce an original-image weight $m$ and three chain weights $w_k$ that sum to one. MetaMix reuses those weights across the two views from each pool. For pool $P$ and view $v$, chain $a^P_{vk}$ contributes to the mixed image

```math
\tilde{x}^{P}_{v}
= m x + (1-m)\sum_{k=1}^{3} w_k\,a^{P}_{vk}(x),
\qquad P\in\{S,U\},\quad v\in\{1,2\}.
```

Let $p_\theta(x)$ be the classifier's predicted class probabilities. Write Equation 6 of the paper using $L_P$ for its objective on pool $P$. The expectation averages over sampled augmentations and examples, and $\lambda$ weights the consistency term.

```math
L_P(\theta,\phi)=\mathbb{E}\left[
\ell_{\mathrm{cls}}(x,y;\theta)+\lambda\,\mathrm{JS}\left(
p_\theta(x),\,p_\theta(\tilde{x}^{P}_{1}),\,p_\theta(\tilde{x}^{P}_{2})
\right)\right].
```

```math
\theta_{\ast}(\phi)\in\arg\min_\theta L_S(\theta,\phi),\qquad
\phi_{\ast}\in\arg\min_\phi L_U\bigl(\theta_{\ast}(\phi),\phi\bigr).
```

The inner objective fits the classifier on seen mixtures. The outer objective scores the resulting classifier on clean images and pseudo-unseen mixtures. The two appearances of $\phi$ in the outer objective matter. Changing the mixer can change **the classifier trained on seen views** and **the unseen mixtures used to evaluate it**. The paper describes a first-order solver. Our rerun implements the full derivative through one classifier step as an explicit interpretation of Algorithm 2. [Wang et al., CVPR 2023, §3.2 and Algorithms 1–3](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_MetaMix_Towards_Corruption-Robust_Continual_Learning_With_Temporally_Self-Adaptive_Data_Transformation_CVPR_2023_paper.pdf#page=5)

### Two limitations motivating the chain-aware runs

**The learned original-image weight rose toward one in our reruns.** In the batch-32 MetaMix run, the epoch mean of $m$ rose from **0.5050** in epoch 1 to **0.9000** in epoch 50. At batch 128 it rose from **0.5028** to **0.9231**. Thus the chains together supplied about 10.0% and 7.7% of the final-epoch blend, respectively. At $m=1$, both mixed views equal the original, so their prediction consistency loss is zero regardless of the chains. This gives the mixer a way to reduce JS by weakening the augmentation. The measured trajectories come from our seed-17 runs. [Batch-32 history](results/unrolled_mixing_seed17/metamix_bs32/progress.json) · [Batch-128 history](results/unrolled_mixing_seed17/metamix/progress.json)

**The MetaMix LSTM assigns weights before observing the sampled chains.** Its inputs summarize the learner and the current and replay images. For a fixed context and recurrent state, resampling the transformations leaves the three output weights unchanged. The weights refer to chain positions, so the mixer can learn a context-dependent preference for a position, but cannot tell which candidate most changed this image's representation. The final-epoch mean of the largest chain weight was 33.46% at batch 32 and 34.24% at batch 128, where an equal three-way split gives 33.33%. These weights divide the augmented share $1-m$. [MetaMix controller](libs/models.py) · [Training loop](reproduce/unrolled_mixing.py)

## Our contribution: chain-aware mixing with independent outer views

We hypothesize that observing how each sampled chain changes the classifier's representation and prediction for an image helps the mixer prioritize chains that benefit training. The mixer learns their relative weights from the classifier step taken on the whole mixture. A shift in weight is favored when that step lowers the sum of two losses: classification error measured by cross-entropy on clean images, and prediction disagreement measured by JS on those images and their independently sampled pseudo-unseen views.

**Learn the weights on seen chains from their effects on the classifier.** For each candidate $a_k$ in a seen view, we compare its feature vector $h_\theta$ and prediction $p_\theta$ with those of the original. A shared scorer $g_\phi$ combines this descriptor with the LSTM's context state $s_\phi$ and assigns relative weights separately for each image and view.

```math
d_k=\mathrm{stopgrad}\!\left(\mathrm{concat}\!\left(
h_\theta(a_k(x))-h_\theta(x),\,
p_\theta(a_k(x))-p_\theta(x)
\right)\right),
\qquad
w^S_k=\frac{\exp g_\phi(s_\phi,d_k)}
{\sum_{j=1}^{3}\exp g_\phi(s_\phi,d_j)}.
```

**Keep the pseudo-unseen evaluation views outside the mixer's control.** Our chain-aware mixer uses changes in the original image's representation and prediction to weight seen chains. If it also weighted pseudo-unseen chains, it could use those signals to favor mixtures that leave the prediction close to the original. This would lower the outer consistency loss by making the evaluation views easier rather than by improving the classifier update. We instead draw the three pseudo-unseen chain weights independently of the mixer using AugMix’s Dirichlet rule, once per image and view, and hold them fixed across the mini-batch’s mixer updates. We fix $m=0.5$ in both pools. The pseudo-unseen views then provide feedback on how the seen mixtures affected the classifier update.

```math
u_v\sim\mathrm{Dirichlet}(1,1,1),\qquad
\tilde x^U_v=\frac{1}{2} x+\frac{1}{2}\sum_{k=1}^{3}u_{vk}\,a^U_{vk}(x).
```

**Train the mixer through one classifier step.** Our inner loss $L_S$ adds seen-view consistency to current-example cross-entropy and DER++ replay (replay cross-entropy weighted by 0.5 and stored-logit MSE by 0.2). The chain-aware experiments use either class-aware InfoNCE on features or JS on predictions. The outer loss $L_U$ always uses clean cross-entropy plus JS on the original and the two pseudo-unseen views.

```math
\theta^+(\phi)=\theta-\alpha\nabla_\theta L_S(\theta,\phi),
\qquad
J_{\mathrm{chain}}(\phi)=L_U\bigl(\theta^+(\phi)\bigr).
```

Because the outer images are independent of $\phi$, the mixer receives feedback entirely through the classifier update:

```math
\nabla_\phi J_{\mathrm{chain}}
=\left(\frac{\partial\theta^+}{\partial\phi}\right)^{\!T}
\nabla_{\theta^+}L_U.
```

This trains the mixer to weight **seen** chains according to how the resulting classifier step affects clean classification and agreement on **pseudo-unseen** views. The operation split and candidate chains are resampled each mini-batch. CIFAR-10-C is reserved for final evaluation. For each of three mixer updates, we recompute the virtual SGD step from the same starting classifier. We commit the classifier step computed with the incoming mixer once per mini-batch. [Training loop](reproduce/unrolled_mixing.py)

**Earlier results used a different gradient.** The historical learned mixers held $\theta^+$ fixed during the outer step and learned weights for the outer views. Their clean classification term contributed zero mixer gradient. Only the direct path through the mixtures remained. Those results are archived below as measurements of that implementation. Our current MetaMix baseline permits both gradient paths.

| Mixer | Original-image weight `m` | How pseudo-unseen views are weighted | How the outer loss reaches the mixer |
| --- | --- | --- | --- |
| Chain-aware | Fixed at 0.5 | Independent Dirichlet(1,1,1) weights per image and view, held fixed for the mini-batch | Through the classifier step on learned seen mixtures |
| MetaMix | Learned | Same learned weights used for seen and pseudo-unseen chains | Through the classifier step and directly through the pseudo-unseen mixtures |
| MetaMix with fixed `m` | Fixed at 0.5 | Same learned chain weights on both pools | Both paths, with a constant original-image contribution |

## Completed one-step experiments

The first table reports seed 17. The runs use the small CNN, a 500-example replay buffer, up to 32 replay examples per update, and ten epochs per task. Class-IL predicts among all ten classes. Task-IL supplies the task's class pair. Each row reports final-checkpoint accuracy (%) on 10,000 clean images and the mean across 15 CIFAR-10-C corruptions and five severities. [Saved configurations and evaluations](results/unrolled_mixing_seed17/)

| Method | Current batch | Seen consistency | Clean class-IL | CIFAR-10-C class-IL | Clean task-IL | CIFAR-10-C task-IL |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Chain-aware | 32 | InfoNCE | 51.36 | 41.23 | 86.90 | 81.04 |
| MetaMix | 32 | JS | 48.89 | 36.67 | 86.47 | 79.45 |
| Chain-aware | 128 | InfoNCE | 39.99 | 32.30 | 83.18 | 76.31 |
| MetaMix | 128 | JS | 37.95 | 28.89 | 83.65 | 77.48 |

At batch 32, chain-aware scored **4.55 percentage points higher** on corrupted class-IL, calculated before rounding. Several choices differ between these rows, including the classifier consistency objective, the inputs to the controller, `m`, and the pseudo-unseen weighting rule. The final-epoch mean largest chain weight was **34.39%** for chain-aware, close to the equal-weight value of 33.33%. The learned MetaMix original-image weight averaged **0.90** in its final epoch. These diagnostics describe the mixtures each model actually used. [Chain-aware progress](results/unrolled_mixing_seed17/chain_random_bs32/progress.json) · [MetaMix progress](results/unrolled_mixing_seed17/metamix_bs32/progress.json)

Batch 32 used constant classifier/mixer learning rates of 0.003/0.005 and 15,650 classifier updates. Batch 128 used 0.012/0.02 and 3,950 updates. Both processed 500,000 current-image presentations. Comparisons across batch sizes therefore also change the learning rates and update count.

### Batch-32 comparison across seeds 17, 23, and 43

Each row averages final accuracy (%) equally across the three seeds. The classifier uses clean cross-entropy and replay in every run. The classifier consistency column specifies the additional loss on pseudo-seen views. All mixers use clean cross-entropy and JS consistency in the outer objective. [Four-method replication runner](scripts/repeat_unrolled_bs32.py) · [MetaMix classifier InfoNCE jobs](jobs/unrolled_bs32/metamix_infonce.pbs)

| Mixer | Classifier consistency | Original-image weight `m` | Clean class-IL | CIFAR-10-C class-IL | Clean task-IL | CIFAR-10-C task-IL |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Chain-aware | JS | 0.5 | 44.59 | 36.56 | 86.26 | 79.14 |
| Chain-aware | InfoNCE | 0.5 | 46.26 | 37.49 | 87.12 | 80.92 |
| MetaMix | JS | 0.5 | 43.51 | 36.38 | 86.89 | 80.05 |
| MetaMix | InfoNCE | 0.5 | 45.71 | 37.27 | 87.12 | 81.01 |
| MetaMix | JS | Learned | 46.16 | 36.43 | 86.65 | 79.83 |
| MetaMix | InfoNCE | Learned | 45.48 | 35.26 | 86.51 | 79.14 |

All six means were computed from the per-seed evaluation files and rounded to two decimals. Differences below use the displayed means and are expressed in percentage points. [Seed-43 evaluation archive](results/unrolled_mixing_seed43/)

1. **Effect of classifier InfoNCE.** Replacing JS with InfoNCE improved all four mean accuracy metrics for both fixed-`m` mixers. On CIFAR-10-C class-IL the gain was 0.93 points for chain-aware and 0.89 for MetaMix. With learned `m`, MetaMix instead lost 1.17 points on that metric.
2. **Effect of chain-aware mixing.** With classifier InfoNCE and `m=0.5` matched, chain-aware led MetaMix by 0.55 points on clean class-IL and 0.22 on CIFAR-10-C class-IL. Corrupted task-IL slightly favored MetaMix (81.01% versus 80.92%). With classifier JS, the corrupted class-IL gap was also small (0.18 points in favor of chain-aware). These comparisons change both candidate scoring and pseudo-unseen weighting, so they cannot isolate the effect of chain descriptors.
3. **Does fixing `m` help MetaMix with classifier InfoNCE?** Yes on average across these seeds. Fixing `m=0.5` raised CIFAR-10-C class-IL from 35.26% to 37.27% (+2.01 points) and CIFAR-10-C task-IL from 79.14% to 81.01% (+1.87 points). Corrupted task-IL improved on all three seeds. The class-IL effect varied by seed, as shown below.

| MetaMix with classifier InfoNCE, CIFAR-10-C class-IL | Seed 17 | Seed 23 | Seed 43 |
| --- | ---: | ---: | ---: |
| Learned `m` | 39.08% | 35.70% | 31.01% |
| Fixed `m=0.5` | 41.32% | 40.50% | 29.98% |
| Fixed minus learned (points) | +2.24 | +4.80 | −1.03 |

### Per-corruption accuracy for the six batch-32 methods

Each entry averages accuracy (%) over five severity levels within each seed, then equally over seeds 17, 23, and 43. On the checkout containing the evaluations, run `python3 scripts/report_unrolled_corruptions.py --update-readme` to populate the Class-IL and Task-IL tables.

<!-- unrolled-corruptions-start -->
<!-- unrolled-corruptions-end -->

## Reproduce a run

From the repository root on a CUDA machine, install the tested PyTorch stack and download the benchmark. The downloader verifies the clean data and the CIFAR-10-C arrays.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python reproduce/download_cifar10_benchmark.py
```

Choose a configuration and pass `--run` to train. Omit that flag to print the command. Use `--data-root /path/to/data` to reuse a verified dataset. The launcher writes checkpoints, progress and evaluation files under the output directory in each configuration.

```sh
python scripts/run_experiment.py configs/unrolled_chain_random_bs32.json --run
python scripts/run_experiment.py configs/unrolled_metamix_bs32.json --run
```

The batch-128 recipes are [chain-aware](configs/unrolled_chain_random.json) and [MetaMix](configs/unrolled_metamix.json). The finite-difference test checks the mixer gradient for all three mixer variants, including which gradient paths survive when the classifier learning rate is set to zero.

```sh
python -m unittest discover -s tests -p test_unrolled_mixing.py
```

## Choosing the mixer objective

Feature MSE trains the mixer to weight seen augmentation chains so their mixtures
approximate pseudo-unseen views in the classifier’s feature space. The [batch-32 comparison](docs/mixer_objective_results.md)
reports clean and corrupted accuracy for feature matching and for learned and random
mixer baselines under classifier InfoNCE.

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

## Earlier experiments

The historical archive includes the detached-classifier runs, random AugMix controls, feature objectives, reconstruction experiments, and scratch ResNet18 comparisons. Earlier references to “corrected mixing” identify revisions to replay, controller context, and BatchNorm handling. The learned mixers in that comparison still used a detached classifier. [Historical evaluations](results/corrected_mixing_seed17/) · [Historical implementation guide](docs/metamix_implementation.md) · [Broader experiment results](docs/all_training_results.md) · [ResNet18 jobs](jobs/resnet18_readme/README.md)

The archived README also records the eight ResNet18 runs and their training settings. [ResNet18 results at the experiment commit](https://github.com/Riezmann75/robust-continual-learning/blob/2e1cf6f47da25bf9899f9962792adb2b46bd7edc/README.md#resnet18-results-seed-17-trained-from-scratch)

<details>
<summary>Historical small-CNN results and corruption tables (seed 17)</summary>

| Mixer | Classifier consistency | Mixer objective | Clean class-IL | CIFAR-10-C class-IL | Clean task-IL | CIFAR-10-C task-IL |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| MetaMix-guided mixer | JS | JS | 46.22% | 37.58% | 89.17% | 83.33% |
| Random mixer | JS | None | 48.80% | 42.11% | 89.71% | 85.41% |
| Random mixer | Class-aware InfoNCE | None | 54.30% | **48.27%** | **91.20%** | **87.19%** |
| MetaMix-guided mixer | Class-aware InfoNCE | Class-aware InfoNCE | 49.42% | 39.68% | 90.08% | 83.24% |
| MetaMix-guided mixer | Class-aware InfoNCE | JS | 48.26% | 38.55% | 89.25% | 82.69% |
| Feature-matching run* | Class-aware InfoNCE | Feature MSE (JS unconfirmed) | **54.96%** | 45.05% | 90.17% | 84.72% |

<!-- core-corruption-ranked-tables -->
All values are **accuracy (%)**, averaged over severities 1-5. **Bold** marks the best and <ins>underline</ins> marks the second-best score in each column, ranked using unrounded values. Objectives are **classifier consistency / mixer objective**; all methods include CE + DER++.

#### Class-IL — all ten classes

| Mixer | Objectives | Noise | Blur | Weather | Digital | Overall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MetaMix-guided mixer | JS / JS | 32.75 | 39.33 | 38.63 | 38.40 | 37.58 |
| Random mixer | JS / none | 40.38 | 44.00 | 41.69 | 41.95 | 42.11 |
| Random mixer | InfoNCE / none | **48.78** | **49.96** | **47.48** | **46.98** | **48.27** |
| MetaMix-guided mixer | InfoNCE / InfoNCE | 36.42 | 40.35 | 40.54 | 40.59 | 39.68 |
| MetaMix-guided mixer | InfoNCE / JS | 34.71 | 40.06 | 39.16 | 39.31 | 38.55 |
| Feature-matching run* | InfoNCE / Feature MSE (JS unconfirmed) | <ins>42.02</ins> | <ins>46.40</ins> | <ins>44.95</ins> | <ins>46.08</ins> | <ins>45.05</ins> |

#### Task-IL — correct class pair supplied

| Mixer | Objectives | Noise | Blur | Weather | Digital | Overall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MetaMix-guided mixer | JS / JS | 81.11 | 82.76 | 85.06 | 83.85 | 83.33 |
| Random mixer | JS / none | 83.22 | <ins>85.60</ins> | <ins>86.44</ins> | <ins>85.84</ins> | <ins>85.41</ins> |
| Random mixer | InfoNCE / none | **86.23** | **87.03** | **88.23** | **87.03** | **87.19** |
| MetaMix-guided mixer | InfoNCE / InfoNCE | 81.53 | 82.69 | 84.75 | 83.57 | 83.24 |
| MetaMix-guided mixer | InfoNCE / JS | 79.01 | 82.57 | 84.74 | 83.52 | 82.69 |
| Feature-matching run* | InfoNCE / Feature MSE (JS unconfirmed) | <ins>83.49</ins> | 84.04 | 86.06 | 84.98 | 84.72 |

<!-- /core-corruption-ranked-tables -->

</details>

The AugMix operations are distributed under Apache 2.0. [Third-party notices](THIRD_PARTY_NOTICES.md) · [AugMix license](reproduce/AUGMIX_LICENSE)
