# MetaMix small-CNN CUDA experiment

This is newly reconstructed code authorized by the user on 2026-11-02, not the
unavailable Mac source and not verified author code. No Mac checkpoint was loaded.
The older metamix_handoff CIFAR-100 experiment is separate and was not substituted.

## Fixed experiment

Seed 17; clean Split-CIFAR10 tasks (0,1), (2,3), (4,5), (6,7), (8,9).
All 50,000 training images, 5,000/class; native 32x32.
100 epochs PER TASK (500 total), 313 batches/epoch including final short batch,
156,500 classifier updates and 5,000,000 presentations.
No random crop/flip, pretraining, residual connections, or test-time augmentation.

Six 3x3 padded stride-1 bias-free convolutions: two each at 32, 64, 128 channels,
each followed by BatchNorm/ReLU. Average pooling 2x2 after stages 1/2.
Global average pooling gives 128 features; linear 10-class head; 288,746 parameters.
Classifier SGD lr .03, momentum=0, weight_decay=0. DER++ alpha=.2 and beta=.5.
Batch32, independent replay CE/distillation samples of up to 32 (without replacement).
Reservoir capacity500 stores raw uint8 CPU images, labels, pre-update logits.
Each repeated presentation counts as a reservoir insertion.

Controller LSTMCell input258/hidden5 and linear output4; zero output head;
initial Gaussian h and c; initial m=.5 and uniform w.
Controller SGD lr .05, J=3. Three chains/view, two views, random depth1-3, severity3.
Per-batch q uniformly sampled in {5,6,7} partitions nine official AugMix operations.
x_mix = m*x + (1-m)*sum(w_i*chain_i(x)); w=softmax(first3), m=sigmoid(last).
Both views share batch-pooled controller weights. m near1 approaches the original.

Detached context: current/replay CE gradient dot product, replay CE, mean current
128 features, mean replay128 features. Empty replay uses zero dot/CE/features.
Auxiliary context passes use train-mode BN with cloned buffers.
Classifier objective: current clean CE + .5 replay CE + .2 replay logit MSE +
JS(original, two pseudo-seen mixed views), coefficient1.
One classifier step, then freeze its UPDATED parameters and use eval-mode committed
BN statistics with cloned functional buffers for three mixer SGD steps minimizing
clean CE + pseudo-unseen JS. Clean CE is constant with respect to mixer parameters.
No second-order differentiation through the classifier update.
Incoming recurrent state reused across all three mixer steps; recompute and advance
once after the last step, detach before the next batch. State persists across tasks.
PIL chains are detached; blending remains differentiable.

## Explicit reconstruction choices and limitations

The paper used ResNet-18 and 50 epochs/task; CNN/100 epochs are user changes.
Batch-pooled context and shared weights simplify per-example alignment.
First-order interpretation, Gaussian initialization, BN convention, severity, and no
extra crop/flip are implementation choices, not a verified reproduction.
Additional details not available from original source are fixed here:
CIFAR normalization mean (.4914,.4822,.4465), std (.2470,.2435,.2616);
gradient dot product is an unnormalized sum over all classifier parameters;
pseudo-seen and pseudo-unseen chains are sampled once per batch and reused for
three mixer steps; after-update JS is measured on those same pseudo-unseen chains.
All BN-affecting auxiliary context forwards use isolated cloned buffers.
Main classifier forwards commit BN statistics sequentially (clean, replay CE,
replay distillation, seen view1, seen view2).
Same seed does not make this code equivalent to the stopped Mac implementation.
One seed and these deviations cannot establish collapse of the authors' code.
A low JS value alone is not robustness evidence; report actual m trajectories and
independent final corruption performance without selecting checkpoints on test data.

## Environment and provenance

Local WSL Ubuntu, NVIDIA RTX3060Ti 8GB; CUDA required with explicit failure otherwise.
Reused existing .venv-nlp virtual environment with system-site-packages from the liver_transplant Conda environment; no dependency modifications. environment_freeze.txt records the installed packages.
Python3.13.11, torch2.10.0+cu128, torchvision0.25.0+cu128, numpy2.4.2,
Pillow12.1.1, matplotlib3.10.8; CUDA runtime12.8, cuDNN91002.
Four CPU threads, cuDNN deterministic enabled, benchmark disabled, TF32 disabled.
Full config records versions, device, all Python source SHA256 hashes.
Checkpoints preserve Python, global NumPy, reservoir-generator, CPU torch and all
CUDA RNG states; model/controller/optimizer/recurrent state and CPU reservoir.
Same-device exact next-step roundtrip was tested. Cross-backend equivalence is not claimed.
Config/source mismatch refuses resume. Use a new output directory for changes.

## Data

Downloader verifies clean archive MD5 c58f30108f718f92721af3b95e74349a via torchvision.
Clean mirror (the Toronto download was slow):
https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python
CIFAR-10-C mirror:
https://huggingface.co/datasets/torch-uncertainty/CIFAR-C/resolve/main/CIFAR-10-C.tar
CIFAR-10-C archive MD5 56bf5dcef84df0e2308c6dcbcbbd8499.
Checks 15 arrays uint8 (50000,32,32,3) and labels equal 5 repetitions of clean test labels.
verified.json records archive checksum and individual file SHA256 hashes.
No corrupted test data enters training or controller optimization.

## Commands (WSL, repository root)

.venv-nlp/bin/python reproduce/download_cifar10_benchmark.py
.venv-nlp/bin/python reproduce/metamix_long.py --device cuda --smoke --output experiments/metamix_cifar10/validation_smoke
.venv-nlp/bin/python tmp/validate_metamix_cuda.py
.venv-nlp/bin/python reproduce/launch_local.py --log experiments/metamix_cifar10/train_cuda.log .venv-nlp/bin/python -u reproduce/metamix_long.py --device cuda

The launcher detaches the Linux job, captures stdout/stderr, and records actual PID
plus /proc start ticks. Host must remain powered on. No remote Mac run is started.
Rerun the same trainer command after interruption: only committed epochs resume.
Uncommitted trace rows are removed before retrying the interrupted epoch.
Never use obsolete Mac PIDs.

## Validation and output

CUDA smoke: parameter count, logits(B,10)/features(B,128), exact m=1 identity,
JS≈0 for identical predictions, initial m/w, finite nonzero mixer gradients with
empty and populated replay. Additional tests check pre-update stored logits,
BN buffer isolation, exact checkpoint next-step roundtrip, reservoir CPU storage,
capacity and repeated-insertion counts, tiny five-task checkpoint/report pipeline,
completed reopen without duplicate traces, malformed/uncommitted trace cleanup,
and config mismatch refusal. Validation-only synthetic runs are separate from training.

Full output: experiments/metamix_cifar10/long_smallcnn_cuda.
MetaMix/checkpoint.pt is atomically replaced after each epoch.
trace.jsonl keeps every batch's m/w before and after, current/replay losses,
seen/unseen JS, all three mixer-step gradient norms and weights.
progress.json reports epoch m mean/min/max, fraction>.95 over BATCHES, not images.
m_progress.png displays the full batch m trajectory and task boundaries.
After all five tasks, evaluate 10,000 clean + 750,000 corrupted examples
(15 corruptions x5 severities x10,000). Both all-ten-output class-IL and
true-class-pair-masked task-IL; no mixer at test. evaluation.json preserves all
corruption/severity results and equal-cell means. complete.json only after evaluation.

## Sources

MetaMix paper:
https://openaccess.thecvf.com/content/CVPR2023/html/Wang_MetaMix_Towards_Corruption-Robust_Continual_Learning_With_Temporally_Self-Adaptive_Data_Transformation_CVPR_2023_paper.html
DER++ reference:
https://github.com/aimagelab/mammoth/blob/master/utils/best_args.py
Official CIFAR-10-C (published checksum verified against this record):
https://zenodo.org/records/2535967
Official Google AugMix operation source and Apache2.0 license retained in
augmix_operations.py and AUGMIX_LICENSE.

## Stopped Mac evidence supplied by user (not locally reproduced)

Task1 epochs1-3 completed:939 updates,30,000 presentations; end m=.738774,
epoch3 mean m=.708473, seen JS=.002043, pseudo-unseen JS=.002320.
No final accuracy exists for that run. It was stopped before this work.
