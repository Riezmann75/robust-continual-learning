# Feature MSE + label-aware InfoNCE + DER++: 0.5 / 0.5 diagnostic

User proposed combining feature MSE and the different-class-negative InfoNCE with
both coefficients reduced to0.5. One fresh seed17 CUDA run,10 epochs/task,50 total.
All prior source/results retained. This is not an optimized or scale-balanced mixture.

## Objective

L=current CE+.5 replay CE+.2 replay-logit mean MSE
  +.5 feature-MSE+.5 label-aware InfoNCE.

Feature-MSE exactly matches the original recipe:
u=h/max(norm(h),1e-8);
.5*mean_examples(sum_coordinates((u0-u1)^2)+sum_coordinates((u0-u2)^2)).
This is128*.5*(ordinary mean-MSE(u0,u1)+ordinary mean-MSE(u0,u2)).
Both original and augmented representations receive gradients.

InfoNCE exactly matches infonce_classneg_cifar10.py:
temperature.1; all3N views are anchors; two same-entry views are positives;
only different-class views are negatives; other same-class views are excluded.
Average log probability of the two positives, with both positives in denominator.
No projection head or extra augmented CE.
A=current batch+CE replay; pooled128-dimensional features before linear head.
Classifier consumes raw features; only consistency losses use normalized features.

Everything else retains the same architecture,normalization,per-image/view AugMix,
RNG streams,shuffle/replay/insertion,SGD.03,batch32,DER++ capacity500,
joint3-view BN forward and separate replay-logit forward.
All50,000 clean training images;15,650 classifier updates;500,000 presentations.
See feature_mse_cifar10.md and infonce_classneg_cifar10.md for detailed source recipes.

## Scale diagnostics

Equal coefficients do NOT imply equal optimization influence.
Every batch logs raw losses,weighted losses,raw encoder gradient norms for each
consistency term,weighted gradient norms,and their gradient cosine.
These diagnostic autograd.grad calls retain the graph but do not accumulate .grad,
run new model forwards,alter BN or consume random draws. Total loss gets one
ordinary backward and one optimizer step. No automatic gradient reweighting.
Extra diagnostic backward work may modestly increase runtime.
Negative counts/no-negative fraction and all existing accuracy/variance diagnostics remain.

## Validation

experiments/mse_infonce_cifar10/validation/validation.json:
Independent loss/mask references,128-dimensional scaling,zero identical-feature MSE,
symmetric finite gradients,weighting and gradient diagnostics,shape/parameter count,
augmentation/RNG isolation,empty/populated replay,short batches,BN counts,
pre-update stored logits,reservoir limits,checkpoint roundtrip,five-task interrupted
and completed resume,trace cleanup,config guard.
At(MSE1,NCE0), model weights/losses exactly recover the original feature-MSE code
in the short test. At(MSE0,NCE1), they exactly recover label-aware InfoNCE.
At both0, they recover the matched DER++ baseline. Existing files unchanged.

## Run

.venv-nlp/bin/python reproduce/mse_infonce_cifar10.py --device cuda --seed 17 --epochs-per-task 10 --lambda-mse .5 --lambda-nce .5 --temperature .1 --output experiments/mse_infonce_cifar10/seed17

Log/PID: experiments/mse_infonce_cifar10/train_seed17.log and train_seed17.process.json.
Source snapshots,config/hardware/package hashes,checkpoints,raw trace,epoch summaries,
diagnostics.png,final evaluation.json and complete.json use the established format.
Final test:10,000 clean+750,000 CIFAR10-C examples,15 corruptions x5 severities,
class-IL(all10 classes) and task-IL(true pair supplied). No test-based selection.
Compare at the same seed; further seeds require a separate request.
This combination also halves the previous NCE coefficient, so any difference
versus NCE1 alone cannot be attributed solely to adding MSE.
