# Matched lambda_feat=0 ablation

User requested exactly one matched ablation of Feature MSE + DER++.
Original: experiments/feature_mse_cifar10/seed17 (lambda_feat=1).
Ablation: experiments/feature_mse_cifar10/seed17_lambda0 (lambda_feat=0).

Fresh seed17; all50,000 CIFAR10 examples;10 epochs/task,50 total;15,650 updates;
500,000 presentations. Same CNN, normalization, optimizer, replay, data order,
independent RNG streams, per-image/per-view Beta/Dirichlet draws, operation chains,
joint original+two-view forward and separate replay-logit forward.
Feature loss is still computed and logged as a diagnostic; only its contribution
to total loss and backward gradient is multiplied by0.
Augmented views remain in the joint BN batch, so this is the matched zero-feature-loss
pipeline rather than a clean-only DER++ baseline. Learned representations, BN running
values and stored logits may diverge as an intended consequence of the loss change.

Only runner changes: explicit --lambda-feat option(default1), validation of nonnegative
finite values, config coefficient, and multiplication in the objective.
Original results/config/source snapshot remain untouched. Config/source guard remains.
Short CUDA comparison showed lambda1 exactly reproduces old code's outputs and weights.
Lambda0 preserved exact original/augmented/replay input tensors, shuffle order,
sampling streams, reservoir images/labels, and BN update counts. Its feature-term
upstream gradient was exactly0. Tiny five-task pipeline/reopen and coefficient guard passed.
Validation evidence: experiments/feature_mse_cifar10/validation_lambda0/validation.json.

Run/resume:
.venv-nlp/bin/python reproduce/feature_mse_cifar10.py --device cuda --epochs-per-task 10 --lambda-feat 0 --output experiments/feature_mse_cifar10/seed17_lambda0

Log: experiments/feature_mse_cifar10/train_seed17_lambda0.log.
PID/start ticks: experiments/feature_mse_cifar10/train_seed17_lambda0.process.json.
Checkpoint,raw traces,progress,plots and full clean/CIFAR10-C evaluation use the same
format as the original. Compare class-IL and task-IL over all10,000 clean and750,000
corrupted examples after completion, with no test-based checkpoint selection.
Single-seed paired evidence isolates the feature penalty within this pipeline;
it does not by itself establish a general or statistically reliable improvement.

Read-only check:
.venv-nlp/bin/python tmp/check_feature_mse_matched.py
This checks the configs differ only in coefficient/source hashes, all training indices
match, and every available ablation batch has exactly the reference sampling diagnostics.
It writes matched_provenance.json and can be rerun after training completes.
