# MetaMix revised first-order experiment

Fresh seed17, CUDA, native32x32, all50000 training images, 10 epochs/task (50 total), batch32/replay32, memory500, same288746-parameter custom CNN, SGD0.03, mixerSGD0.05, J3. No old checkpoint resumed. Compare to the original local MetaMix-inspired result; this is not a verified reproduction of author code.

Corrections based on Algorithm3 and supplement Table4:
- Current and replay CE batches are concatenated before seen/unseen augmentation and JS. Distillation uses its independent replay sample as before. Store only current pre-update logits; reservoir counts repeated presentations.
- Individual 258-dimensional contexts contain the batch gradient dot product and replay CE broadcast to each row plus individual replay/current128features in paper order er,et. Context is detached.
- State is64x5 for hidden and cell, with fixed current slots0:32 and replay32:64. Each row produces its own sigmoid m and softmax three-chain w. The same example's weights are shared by its two views.
- Save state produced by incoming phi BEFORE the solver. Each of J mixer updates reuses incoming state. Post-update output recomputation is diagnostic only. Detach between batches/tasks; inactive slots retain state.

Unspecified conventions explicitly chosen: cyclic positional pairing of current and replay features, no feature averaging. Current row i uses er[i % replay_count],et[i]; replay row j uses er[j],et[j % current_count]. Empty replay er and replay scalars are zeros. Slot histories follow batch positions, not persistent image identity. Gaussian initial states; zero mixer output head retained. No full temporal BPTT. No crop/flip; severity3. All are implementation choices, not established author details.

Solver: retain Algorithm2's explicit first-order reading. Update classifier once with original current CE +0.5replayCE +0.2logitMSE + mean JS across current+replay. Then hold updated classifier fixed, take three phi steps on original concatenated-batch CE + unseen JS. Clean CE has exactly zero phi gradient in this interpretation. No second-order unrolling and no augmented CE added. The requested gradient-fidelity issue is not claimed resolved: paper allows first-order but does not specify autograd details sufficient to certify this as author behavior. This run cannot settle whether the authors' code has the identity shortcut.

BN: context train forwards use cloned buffers. Joint current/replay original forward and both seen forwards commit training BN; independent distillation forward also commits BN. Outer passes use eval-mode committed BN with isolated buffers. Joint clean forward is a consequence of concatenating the batch, and differs from the older separate current/replay clean passes.

Checkpoints include 64-slot state, both optimizers, model/controller, reservoir, RNG, and config with source hashes. Immutable source snapshots accompany the run. Trace includes per-example m/w before/after, all mixer steps, current/replay JS diagnostics. Progress reports batch means and separately per-example statistics (including replay repetitions). Final eval covers10000clean +750000CIFAR10-C samples, class-IL and true-pair task-IL, no test-based selection. Model unchanged at inference.

CUDA tests: per-example blend/identity; current+replay augmentation shapes; individual feature context; pre-solver state timing and inactive slots; empty/populated and short batches; finite gradients; explicit CE gradient contract; exact next-step resume; interrupted five-task pipeline and no duplicate records on completed reopen.
