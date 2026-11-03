# Fresh MetaMix joint-view BatchNorm run

User requested stopping the separate-forward revised run and restarting on 2026-11-02. New output experiments/metamix_jointbn/seed17; fresh seed17, 10epochs/task, all50000 images, CUDA float32. The previous run is preserved under experiments/metamix_revised/seed17 with stopped.json (29 committed epochs).

All scientific settings inherit docs/metamix_revised.md except the explicit joint-view forward change:
- Classifier: concatenate original current+CE-replay and its two pseudo-seen mixed views into one training forward. Split outputs for unchanged clean CE/DER++ and JS objectives. Capture original current logits before the optimizer step.
- Distillation: independent replay sample, separate training forward after the joint forward. Therefore BN increments once when replay empty, twice when replay exists.
- Mixer: concatenate original current+CE-replay and two pseudo-unseen mixed views into one eval-mode functional forward for each of three mixer updates. No committed BN mutation. Diagnostic post-update joint forward also eval mode. No additional standalone clean forward.
- Context auxiliary forwards remain training-mode with cloned BN buffers; they do not commit running statistics.

Per-example mixing,64persistent batch-slot states,pre-solver state advancement,cyclic feature pairing, DER++ coefficients, SGD and both learning rates unchanged. No augmented CE or second-order differentiation. The clean CE contribution to mixer gradients remains zero under the documented first-order interpretation; author-code fidelity unresolved. This is a user-requested BN variant, not a claim that the paper specifies joint BN.

Tests: inherited replay/individual-state/gradient/resume checks; exact train/eval forward batch sizes; committed BN update counts; no outer buffer mutation; joint-vs-separate eval equivalence; five-task checkpoint/plot/no-duplicate reopening. Final eval10000clean +750000corrupted images without augmentation. No other experiments launched.
