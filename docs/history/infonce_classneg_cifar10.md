# Label-aware InfoNCE + DER++: only different-class negatives

User requested changing standard InfoNCE to use other classes as negatives.
Fresh seed17 run;10 epochs/task,50 total. Standard InfoNCE and prior outputs retained.
Same temperature.1,lambda_nce1,CNN,SGD.03,batch32,DER++ alpha.2/beta.5,capacity500,
full50,000 clean images,random views,RNG streams,normalization and BN forwards.
See infonce_cifar10.md for the shared recipe.

## Sole scientific change: negative mask

A=B concatenated with replay CE examples. There are3N anchors from original plus
two AugMix views. Each anchor has TWO positives: other views of its own batch entry.
Only views of DIFFERENT-CLASS entries are negatives.
Other same-class entries are EXCLUDED from the denominator, not additional positives.
Self-comparisons are also excluded. Labels from current training and replay define
the mask; no test labels or corrupted-test images enter training.
Repeated same-class source images in separate entries are consequently excluded.

For each anchor:
L_i=logsumexp(scores over two positives and different-class views)
    -mean(scores of its two positives).
scores=cosine(normalized128 features)/.1.
Mean over all3N anchors; symmetric gradients; no projection head.
Current CE+.5 replay CE+.2 replay-logit MSE+1*L_InfoNCE.
Current/replay originals and both views still use one joint BN forward;
distillation replay has the same separate BN forward.

Eligible negative counts vary with class composition. Log their mean,min,max and
fraction of anchors with no negatives. If no different-class entries exist, retain
both positives and compute their finite two-candidate loss (log2 if tied).
Zero is logged for negative cosine when no negatives exist; negative-count
diagnostics distinguish that sentinel. Retrieval accuracy is over eligible candidates.
Its value and raw InfoNCE loss are not directly comparable with the standard run's
larger candidate pool. This is not the standard unconditional MI lower-bound setup.

## Validation

CUDA checks passed: independent anchor-loop reference, correct exclusion of all
same-class other entries, two retained positives, all-unique-class equivalence with
standard InfoNCE,no-negative and one-example edge cases,finite symmetric gradients,
encoder/head gradient routing,exact copied CNN and shapes,contiguous tensors,
random augmentation/RNG isolation,empty/populated replay,short batch,BN counts,
pre-update logits,reservoir limits,exact next-step checkpoint roundtrip,
five-task interrupted/completed resume,trace cleanup and config guard.
At lambda0, this code exactly reproduces the matched DER++ baseline's weights and
forward inputs in the short test.
Evidence: experiments/infonce_classneg_cifar10/validation/validation.json.

## Execution

.venv-nlp/bin/python reproduce/infonce_classneg_cifar10.py --device cuda --seed 17 --epochs-per-task 10 --lambda-nce 1 --temperature .1 --output experiments/infonce_classneg_cifar10/seed17

Log/PID metadata: experiments/infonce_classneg_cifar10/train_seed17.log and
train_seed17.process.json. Output includes checkpoint.pt,config.json,source snapshot,
raw traces,progress.json and diagnostics.png. Source/config guards remain.
Final evaluation:10,000 clean test +750,000 CIFAR10-C examples (15x5x10,000),
all-ten-class and true-class-pair-masked accuracy; no test-based selection.
Single-seed diagnostic; no extra runs launched. Candidate masking changes the loss
and gradient scale even with the same coefficient; no optimum is claimed.
