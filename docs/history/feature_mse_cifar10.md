# Feature MSE + DER++ (random AugMix views)

Implemented from the user's proposed recipe, with their explicit subsequent change
to **10 epochs per task (50 total)**. Fresh seed17 initialization; no MetaMix weights.
This is a proposed experiment, not a MetaMix-paper method or a claim of improvement.

## Budget and network

All50,000 clean CIFAR10 train images, tasks(0,1),(2,3),(4,5),(6,7),(8,9), shuffled
within each epoch; native32x32. Batch32 including short batch16:313 updates/epoch.
Total15,650 classifier updates and500,000 current presentations. Constant SGD.03,
momentum0, weight_decay0. Reservoir500; independent CE and distillation samples
up to32 without replacement, selected before current insertion. DER++ beta.5/alpha.2.

SmallCNN architecture copied unchanged from reproduce/metamix_long.py, verified by AST:
six bias-free3x3 convolutions (32,32,64,64,128,128), BN/ReLU, average pooling after
first two stages, global average pooling,128-feature linear10-class head.
288,746 parameters; no pretraining/residuals.
Network wrapper normalizes pixels internally: mean(.4914,.4822,.4465),
std(.247,.243,.261), exactly as the new recipe specifies.
This differs from the old local runner's std(.2470,.2435,.2616).
The classifier consumes raw pooled features; only the consistency loss L2-normalizes.

## Views and objective

A=current batch concatenated with CE replay. Independently for each image and each
of two freshly sampled views: three chains, depth uniformly1-3, nine official
AugMix PIL operations with replacement, severity3. No extra crop/flip/noise.
w~Dirichlet(1,1,1),m~Beta(1,1), independently per image/view.
view=m*original+(1-m)*sum(w_k*chain_k), in[0,1] pixels before normalization.
No learned weights, controller, teacher, projector, predictor, JS, OT or outer loop.
PIL augmentation and mixing have no autograd; all encoder forwards do.

u=h/max(norm(h),1e-8). Feature loss is .5*mean_examples(
sum_coordinates((u0-u1)^2)+sum_coordinates((u0-u2)^2)).
This equals128*.5*(mean_MSE(u0,u1)+mean_MSE(u0,u2)).
Both original and augmented branches get gradients. lambda_feat=1.
Total=current clean CE+.5*replay clean CE+.2*ordinary logit MSE+feature loss.
No augmented CE. All A examples have equal weight in feature loss.
Joint forward on concat(original,view1,view2) updates BN once. Separate original
distillation replay forward updates BN again when replay exists.
Current logits from joint forward are detached/stored before the optimizer update.
Only current presentations enter reservoir; repeated epochs count again.

Independent seeded RNG streams: insertion,replay_ce,replay_logit,shuffle,mixture,
operations. The official operations' np.random calls run within a single-threaded
scope that swaps/restores their dedicated RandomState without changing global state.
All streams plus Python/NumPy/torch CPU/device RNGs are saved/restored.

## Validation and provenance

Passed on RTX3060Ti CUDA: copied architecture and count/shapes, contiguous float32,
weight simplex/range/independence/no-grad, RNG isolation, exact m=1 identity,
zero loss for identical features,128x scaling, symmetric gradients, feature-only
encoder gradients and CE head gradients, empty/populated replay, short batch,
BN update counts, detached pre-update logits, reservoir capacity/presentations,
exact CUDA next-step checkpoint roundtrip, five-task intermediate resume,
completed reopen without duplicated rows, uncommitted-trace cleanup, config guard.
Results: experiments/feature_mse_cifar10/validation/validation.json.
Validation synthetic data is never used for the full run.

Device explicitly cuda/mps/cpu with availability checks, no fallback. CUDA is used.
MPS/CPU paths are supported but not validated on this machine.
Reused .venv-nlp (inherits liver_transplant Conda packages); no dependencies changed.
Python3.13.11,PyTorch2.10.0+cu128,torchvision0.25.0+cu128,float32.
GPU/software/source hashes/data verification SHA recorded in config.json.
Source and Apache2.0 AugMix license snapshots saved under source/.
All existing MetaMix source files and results preserved.
Old and new runs differ in views, BN convention, normalization and loss; they are
not a controlled loss-only comparison. One seed cannot establish general benefit.
Fixed augmentation removes learned-m weakening but does not guarantee robustness
or prevent all representation degeneracies.

## Active run and outputs

.venv-nlp/bin/python reproduce/feature_mse_cifar10.py --device cuda --epochs-per-task 10 --output experiments/feature_mse_cifar10/seed17

Detached launcher metadata: experiments/feature_mse_cifar10/train_seed17.process.json.
Log: experiments/feature_mse_cifar10/train_seed17.log.
Same command resumes compatible checkpoint at the next epoch; partial epochs rerun.
checkpoint.pt atomically saves/fsyncs model,optimizer,CPU reservoir,RNG,config,epoch
position and epoch summaries. Config/source mismatch refuses resume.
trace.jsonl contains batch loss/accuracy/cosine/norm/variance/gradient/sample-weight
diagnostics. progress.json holds epoch summaries, diagnostics.png plots trajectories.
Epoch metric summaries average batches; sampling_m pools all image/view samples.
m statistics are random distribution diagnostics, not learned trajectories.
No test-driven checkpoint/lambda selection.

After task5, full eval mode/no augmentation:10,000 clean and750,000 corrupted
examples (15 standard CIFAR10-C corruptions x5 severities).
evaluation.json contains per-cell class-IL(all10) and task-IL(true-pair mask)
accuracies and equal-cell averages. complete.json only after successful evaluation.
Data root: experiments/metamix_cifar10/data. Existing data hashes reverified.
