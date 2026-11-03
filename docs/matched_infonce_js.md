# Corrected controller: seen InfoNCE, unseen JS

This fresh seed-17 CUDA run changes only the learned mixer's pseudo-unseen consistency objective from InfoNCE to JS relative to the completed corrected learned-InfoNCE run. Classifier objective remains current clean CE + 0.5 replay CE + 0.2 logit MSE + class-aware InfoNCE, weight 1 and temperature 0.1. Mixer objective is clean CE plus JS, weight 1, with the updated classifier fixed; clean CE has no mixer gradient. J=3, mixer SGD 0.05. No hypergradient or explicit retention outer objective is added.

The original corrected Controller is copied unchanged: LSTMCell(258,5), Linear(5,4), learned sigmoid m and softmax three-chain w, shared across the two views per example. There are no candidate-chain inputs and m is not fixed. Preserve context pairing, 64 recurrent slots, pre-solver state advancement, joint three-view BatchNorm, independent logit replay, and seen/unseen operation partitions. Classifier, controller initialization and sampling streams match the reference. See [paper-to-code guide](metamix_implementation.md) for unresolved fidelity questions.

Use all 50,000 clean images at 32x32, five class-pair tasks, 10 epochs/task, batch 32 including short batches, reservoir 500, classifier SGD 0.03. Total 15,650 updates and 500,000 presentations. Full final evaluation uses 10,000 clean plus 750,000 CIFAR-10-C images, class-IL and true-pair task-IL. No test-based checkpoint selection. Fresh output: experiments/matched_mixing/learned_infonce_js/seed17.

```sh
python scripts/run_experiment.py configs/matched_infonce_js.json --data-root /path/to/data --run
python -m unittest discover -s tests -p test_matched_infonce_js.py -v
```

Validation checks exact equivalence to the reference when outer InfoNCE is retained, and exact equivalence when only reference outer_objective is switched to JS. Also tests replay/short batches, nonzero gradients for learnable m, finite gradients, BatchNorm counts, exact next-step resume, and tiny five-task interrupted/completed checkpoint reopen without duplicate trace records.


## Completed result

Seed 17 completed all 15,650 updates and the full evaluation. Clean class-IL: **48.26%**; CIFAR-10-C class-IL: **38.55%**; clean task-IL: **89.25%**; CIFAR-10-C task-IL: **82.69%**. Final-epoch mean m: **0.9738855**, with **99.2606%** of per-example values above 0.95. All sampling fingerprints match the preceding matched runs. This single-seed outer-loss change did not improve over InfoNCE/InfoNCE (49.42% clean, 39.68% corrupted class-IL). [Full results and exact source snapshot](../results/corrected_mixing_seed17/learned_infonce_js/).
