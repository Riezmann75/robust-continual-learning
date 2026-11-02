# Two additional matched seeds

Authorized seeds18 and19, each with lambda_feat1 and0. Four additional runs,
10 epochs/task,50 total per run, all50,000 clean training examples. Sequential CUDA
queue avoids concurrent GPU contention. Existing seed17 results are preserved.

Order: seed18 lambda1, seed18 lambda0, seed19 lambda1, seed19 lambda0.
Outputs: experiments/feature_mse_cifar10/seed18, seed18_lambda0, seed19, seed19_lambda0.
Each starts fresh. The seed controls model initialization and all independent
shuffle, insertion, replay, operation and mixture RNG streams.
Only source change: --seed CLI(default17), range validation, and routing that
value to setup, Streams, and config. Seed17 exact-behavior regression passed.

The tiny five-task validations exercise each new seed/coefficient pair.
Pair audits require identical train indices, sampling diagnostics for every batch,
final RNG states and final reservoir images/labels. Configs differ only in lambda.
Learned weights, stored logits and BN running values may differ by design.

Launch queue:
.venv-nlp/bin/python reproduce/launch_local.py --log experiments/feature_mse_cifar10/additional_seeds_queue.log .venv-nlp/bin/python -u tmp/run_feature_mse_seeds.py

Queue state and actual child PIDs:
experiments/feature_mse_cifar10/additional_seeds/queue_status.json
Individual logs:
experiments/feature_mse_cifar10/additional_seeds/seed{18,19}_lambda{0,1}.log
The queue stops on training/audit failure and records the error. Source hashes are
checked before each job. Compatible interrupted jobs resume from checkpoints.

After all four runs finish, three_seed_comparison.json in experiments/feature_mse_cifar10
contains per-seed clean/CIFAR10-C class-IL/task-IL accuracies for seeds17,18,19,
paired lambda1-lambda0 differences, and means/sample standard deviations.
Accuracy values are fractions; multiply by100 for percentages or percentage-point
differences. Three seeds provide a limited replication, not a broad generalization.
All runs evaluate the same full10,000 clean and750,000 corrupted examples.
No test-based checkpoint/lambda selection.
