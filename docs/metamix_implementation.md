# MetaMix: paper-to-code implementation guide

This file restates how we implemented the MetaMix paper in this repository. It
maps the paper's equations and algorithms to our code, explains one training
step, and separates implemented mechanisms from our choices and unresolved
fidelity questions. It describes the corrected **learned-mixer JS** reference;
the other three comparison runs are extensions or controls, not paper reproductions.

## Sources and code

- [MetaMix, CVPR 2023: main paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_MetaMix_Towards_Corruption-Robust_Continual_Learning_With_Temporally_Self-Adaptive_Data_Transformation_CVPR_2023_paper.pdf): Section 3.2, Eqs. (3)-(8), Algorithms 1-3.
- [Supplement](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Wang_MetaMix_Towards_Corruption-Robust_Continual_Learning_With_Temporally_Self-Adaptive_Data_Transformation_CVPR_2023_supplemental.pdf): implementation settings and notation, including Table 4.
- [Corrected JS runner](../reproduce/metamix_jointbn.py), [configuration](../configs/metamix_jointbn.json), and [CUDA tests](../tests/test_metamix_jointbn.py).
- [Shared comparison runner](../reproduce/matched_mixing.py) and [comparison details](matched_mixing.md).
- [Exact completed-run sources and evaluations](../results/corrected_mixing_seed17/).

We implemented from the paper and supplement, not verified author source code.
Matching a listed algorithmic component does not establish full reproduction.

## How the paper maps to the code

| Paper component | Code | Implementation and qualification |
|---|---|---|
| Context, Eq. (5) | `context_for` | Clean current/replay gradient dot product, replay CE, and individual replay/current features. Feature pairing and scalar broadcasting are our conventions. |
| Temporal mixer, Eqs. (3)-(4) | `Controller`, `initial_state` | LSTM hidden size 5; three softmax chain weights and one sigmoid original-image weight per example. Persistent batch-slot interpretation is not verified against author code. |
| Data mixing, Algorithm 1 | `chains`, `blend` | Two views, three randomly sampled chains per view; mix their outputs and the original image. |
| Current/replay concatenation, Algorithm 3 | `step` | Augment current examples and replay-CE examples together. Logit-distillation replay is sampled independently for DER++. |
| Pseudo-seen/unseen split, Eq. (6), Algorithm 3 | `step` | Split nine operations each batch into disjoint pools of 5-7 and 2-4 operations. |
| Classifier then mixer updates, Algorithm 2 | `step`, `outer_objective` | One classifier step followed by three mixer steps with the updated classifier fixed. Exact gradient fidelity remains unresolved. |
| Prediction consistency, Eq. (8) | `js` | Mean KL divergence from each of the three prediction distributions to their mean. |
| Recurrent state timing, Algorithm 3 | `step` | Commit the transition computed before mixer optimization; reuse the incoming state for its three optimization steps. |

## What the LSTM receives

For each example, the context has 258 coordinates:

1. One batch-level dot product between current and replay clean-CE gradients with respect to CNN parameters.
2. One batch-level replay clean-CE loss.
3. A 128-dimensional replay feature vector.
4. A 128-dimensional current feature vector.

Both scalar values are broadcast across the examples. Gradients are computed
without taking an optimizer step; their dot product indicates local agreement or
conflict between the two classification objectives. Context is detached before
entering the controller. It does not contain operation identities, augmented
features, or candidate-chain scores.

For current row i, replay features are paired cyclically by i modulo replay
batch size; replay rows use the analogous pairing with current features. Empty
replay gives zero replay features and zero replay-related scalars. This pairing
is an explicit implementation choice, not a specified paper detail.

Hidden and cell states each have 64 slots of width 5: 32 current and 32 replay
positions. Slots follow batch positions, not persistent image identities.
Inactive slots are retained; states are detached between batches and tasks.
Gaussian initial recurrent states and a zero output head give initial m=0.5 and
uniform chain weights. These initialization and truncation conventions are ours.

## One training step

1. Sample a current clean batch and two independent replay batches: one for CE,
   one for stored-logit distillation. Concatenate current and replay-CE examples
   into A. Compute the clean context.
2. Run the LSTM on that context and incoming state to obtain per-example m and w.
   Save this pre-solver state transition. Both augmented views share each
   example's weights. Candidate operations have not been supplied to the LSTM.
3. Split the operation pool and generate two pseudo-seen and two pseudo-unseen
   views of A. Each view has three chains, depth sampled from 1-3 with replacement
   over operations and severity 3. Use the included licensed AugMix PIL operations;
   do not add crop/flip. Chains are detached, but blending is differentiable:

   `mixed = m * original + (1-m) * sum_k(w[k] * chain[k])`

4. Update the classifier with:

   `current CE + 0.5 * replay CE + 0.2 * replay-logit MSE + seen JS`

   CE applies to original images. JS averages across current and replay-CE
   examples. Logit MSE averages over examples and all ten coordinates. Capture
   current logits before the update for later storage.
5. Hold the updated classifier fixed. Take three mixer SGD steps, learning rate
   0.05, on original-A CE plus pseudo-unseen JS. Recompute weights with the same
   detached context and incoming recurrent state at each step. The sampled
   pseudo-unseen chains are reused across these steps.
6. Keep the recurrent state saved in step 2. Insert only current raw images,
   labels and pre-update logits into the reservoir. It counts every presentation,
   including repeated examples across epochs; replay samples are not reinserted.

Pseudo-unseen means excluded from the classifier's augmentation pool for that
batch. These operations are resampled into new pools next batch; they are not
permanently held out and are not CIFAR-10-C test corruptions. The intended purpose
is to train the mixer using transformations different from those just used to
update the classifier.

## Gradient interpretation: the main unresolved issue

The paper presents a bilevel objective and describes a first-order solver similar
to first-order MAML. Our implementation uses an explicit alternating reading:
classifier parameters are detached after its update, and there is no derivative
through that update when training the mixer. Original-image predictions in the
outer objective are detached as well.

Consequently, outer clean CE has **zero mixer gradient** in this implementation;
JS provides the mixer gradient through image blending and the fixed classifier.
We have tested that behavior, but have not established that it matches the
original authors' intended autograd implementation. We therefore do not claim
that this fidelity question was fixed by the other corrections.

The controller can reduce direct consistency loss by increasing m toward 1.
Our observed high m values describe this implementation. They do not prove that
verified author code collapses, or that low JS implies corruption robustness.

## BatchNorm and experimental differences

Our user-requested BatchNorm convention is one joint training forward for the
original and two pseudo-seen mixed views, followed by a separate replay-logit
forward when replay exists. Context forwards use training statistics with cloned
buffers, so they do not commit running-statistic updates. Each outer forward
joins original and two pseudo-unseen views in eval mode using the updated
classifier's committed statistics and isolated functional buffers. This is not
claimed to be the paper's exact BatchNorm convention.

Other differences and explicit choices:

- Custom six-convolution CNN, 288,746 parameters and 128 pooled features, rather
  than the paper's ResNet-18. No residual connections or pretraining.
- 10 epochs/task and seed 17 for this comparison, rather than the paper's
  50 epochs/task and multiple runs. Native CIFAR 32x32 input; this is our default,
  not a claim that the paper explicitly states every input resolution.
- All 50,000 training images; five sequential class pairs; batch 32, reservoir
  500; classifier SGD 0.03 without momentum or weight decay; constant schedule.
- Severity, initialization, feature pairing, recurrent batch slots, and the
  absence of additional crop/flip are disclosed conventions.

## How the four corrected experiments differ

| Mixer | Classifier consistency | Mixer consistency |
|---|---|---|
| Learned | JS | Pseudo-unseen JS |
| Random | JS | None; unseen views only provide diagnostics |
| Random | Class-aware InfoNCE | None; unseen views only provide diagnostics |
| Learned | Class-aware InfoNCE | Pseudo-unseen class-aware InfoNCE |

InfoNCE is our replacement objective, not part of the MetaMix paper. It uses
normalized pooled features, temperature 0.1, same-image positives across three
views and only different-class negatives. Random mixtures sample Beta(1,1) m and
Dirichlet(1,1,1) w independently per image and view. Learned weights are shared
across the two views, so the comparison is not an isolated mixer-only ablation.
The later chain-aware, fixed-m controller is a separate proposed-method experiment.

## Validation and evaluation

CUDA tests cover per-example blending, identity mixing, context shapes,
current/replay augmentation, state timing, empty/populated replay and short
batches, finite gradients, the outer CE gradient contract, BatchNorm behavior,
exact next-step resume, and tiny five-task checkpoint/plot runs. Completed
checkpoint reopening does not duplicate logs. These tests validate our stated
implementation, not equivalence to unavailable author code.

All four completed runs use final-checkpoint evaluation on 10,000 clean test
images and 750,000 CIFAR-10-C images: 15 corruptions at five severities. Class-IL
uses all ten outputs; task-IL is given the correct class pair. No mixing is used
at test, and test results do not select checkpoints. See the
[results table and limitations](../README.md#corrected-mixing-comparison-seed-17).
