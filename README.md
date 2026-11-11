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

**Evaluate the classifier on mixtures sampled independently of the mixer.** We fix $m=0.5$ in both pools. The learned weights combine seen chains. For each image and each pseudo-unseen view, we draw three fresh chain weights using AugMix's Dirichlet rule and hold that draw fixed throughout the mini-batch's mixer updates.

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

All runs below use seed 17, the small CNN, a 500-example replay buffer, up to 32 replay examples per update, and ten epochs per task. Class-IL predicts among all ten classes. Task-IL supplies the task's class pair. Each row reports final-checkpoint accuracy (%) on 10,000 clean images and the mean across 15 CIFAR-10-C corruptions and five severities. [Saved configurations and evaluations](results/unrolled_mixing_seed17/)

| Method | Current batch | Seen consistency | Clean class-IL | CIFAR-10-C class-IL | Clean task-IL | CIFAR-10-C task-IL |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Chain-aware | 32 | InfoNCE | 51.36 | 41.23 | 86.90 | 81.04 |
| MetaMix | 32 | JS | 48.89 | 36.67 | 86.47 | 79.45 |
| Chain-aware | 128 | InfoNCE | 39.99 | 32.30 | 83.18 | 76.31 |
| MetaMix | 128 | JS | 37.95 | 28.89 | 83.65 | 77.48 |

At batch 32, chain-aware scored **4.55 percentage points higher** on corrupted class-IL, calculated before rounding. Several choices differ between these rows, including the classifier consistency objective, the inputs to the controller, `m`, and the pseudo-unseen weighting rule. The final-epoch mean largest chain weight was **34.39%** for chain-aware, close to the equal-weight value of 33.33%. The learned MetaMix original-image weight averaged **0.90** in its final epoch. These diagnostics describe the mixtures each model actually used. [Chain-aware progress](results/unrolled_mixing_seed17/chain_random_bs32/progress.json) · [MetaMix progress](results/unrolled_mixing_seed17/metamix_bs32/progress.json)

Batch 32 used constant classifier/mixer learning rates of 0.003/0.005 and 15,650 classifier updates. Batch 128 used 0.012/0.02 and 3,950 updates. Both processed 500,000 current-image presentations. Comparisons across batch sizes therefore also change the learning rates and update count.

### Follow-up comparisons at batch 32

| Run | Question | Status as of Nov 11, 2026 |
| --- | --- | --- |
| [Chain-aware with seen JS](configs/unrolled_chain_random_js_bs32.json) | How much does replacing InfoNCE with MetaMix's classifier consistency objective change accuracy? | Training |
| [MetaMix with fixed m](configs/unrolled_metamix_fixed_m_bs32.json) | How much does fixing the original-image weight at 0.5 change MetaMix when it still learns the chain weights? | Queued behind chain-aware JS |

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
| Chain-aware mixer (`m = 0.5`) | Class-aware InfoNCE | JS | **56.22%** | 47.17% | 91.02% | 86.24% |
| Chain-aware mixer (`m = 0.5`) | JS | JS | 50.33% | 42.37% | 90.50% | 85.66% |
| Chain-aware mixer (`m = 0.5`) | Class-aware InfoNCE | Class-aware InfoNCE | 50.15% | 42.79% | 90.99% | 86.01% |

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

<!-- /core-corruption-ranked-tables -->

</details>

The AugMix operations are distributed under Apache 2.0. [Third-party notices](THIRD_PARTY_NOTICES.md) · [AugMix license](reproduce/AUGMIX_LICENSE)
