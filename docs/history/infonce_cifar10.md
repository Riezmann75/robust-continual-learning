# InfoNCE + DER++ (random AugMix views), seed17 diagnostic

User authorized replacing feature MSE with standard InfoNCE while retaining current
classification CE and DER++ replay. Same-class images are negatives. One initial
seed17 run, fresh initialization,10 epochs/task (50 total), all50,000 clean images.
Prior MetaMix and feature-MSE runs are preserved; no prior weights are loaded.

## Exact contrastive objective

A is current B concatenated with R_ce. N=len(A). Three aligned views: original,
AugMix1,AugMix2. Same original architecture and normalized128-dimensional pooled
features; no projection head. Classifier sees unnormalized pooled features.
All3N views serve as anchors. Anchor i has the other two views of the same batch
entry as positives. Every view of every OTHER batch entry is a negative, including
same-class examples. Labels never enter the contrastive loss or masks.
Self-similarity is excluded from the denominator.

s_ij=cosine(h_i,h_j)/temperature.
L_NCE=mean_i [logsumexp_{k!=i}(s_ik) - mean_{p in positives(i)} s_ip].
This averages the log probabilities of two positives, not the log of their summed
probabilities. Denominator includes both positives and all negatives.
All encoder branches receive gradients. Stable logsumexp is used.
Two positives means identical tied features give log(3N-1), not zero loss;
even perfectly isolated identical positive triplets have limiting loss log2.
Do not compare this loss's numerical scale directly to MSE or interpret it as a
measured mutual-information value.

Temperature=.1 and lambda_nce=1 are predeclared diagnostic starting settings,
not tuned on test data and not claimed to match lambda_mse=1 in gradient strength.
Total=current CE+.5 replay CE+.2 replay-logit mean MSE+lambda_nce*L_NCE.
Only feature MSE is replaced; DER++'s replay-logit MSE remains ordinary mean MSE.
No augmented CE, learned mixer, LSTM, JS, teacher or extra projection layers.

Batch-entry identity is the positive key. As in ordinary instance-contrastive
batch construction, separate entries are not deduplicated: repeated source images
arising from repeated reservoir presentations may be treated as negatives.
This is disclosed rather than silently changing replay sampling.

## Matched training protocol

Same feature-MSE CNN:288,746 parameters; channels32/64/128, two conv/BN/ReLU each,
average pooling after stages1/2, global pooling and linear10-class head.
Normalization mean(.4914,.4822,.4465),std(.247,.243,.261).
Native32x32; no crop/flip/pretraining. Random views exactly as previous experiment:
two views,3 PIL chains/view, depth1-3,severity3,9 AugMix operations, independent
per-image/view w~Dirichlet(1,1,1),m~Beta(1,1).
Current batch32 with final short batch16; reservoir500; independent CE/logit replay
batches up to32. Separate shuffle/insertion/replay/operation/mixture RNG streams.
Joint3-view forward updates BN once; replay-logit forward updates BN once more.
Pre-update current logits saved in CPU reservoir; only current examples inserted.
Constant SGD lr.03, momentum0, weight_decay0, float32,CUDA.
15,650 classifier updates and500,000 current presentations.

## Validation

experiments/infonce_cifar10/validation/validation.json:
Architecture/shapes,contiguous tensors,random weights and RNG isolation,identity blend,
independent reference InfoNCE formula,self/positive masks,known equal-feature loss,
matched vs shuffled positives,symmetric gradients,empty/populated replay,short batch,
BN update counts,pre-update stored logits,reservoir limits,exact same-CUDA next-step
checkpoint restoration,five-task intermediate resume,no duplicate rows,trace cleanup,
config mismatch refusal. At coefficient0, new code exactly reproduced the previous
lambda_feat0 model state and input tensors in the short test.

## Run and provenance

.venv-nlp/bin/python reproduce/infonce_cifar10.py --device cuda --seed 17 --epochs-per-task 10 --lambda-nce 1 --temperature .1 --output experiments/infonce_cifar10/seed17

Log: experiments/infonce_cifar10/train_seed17.log
Actual PID/start ticks: experiments/infonce_cifar10/train_seed17.process.json
Outputs: checkpoint.pt,config.json,source/,train_indices.json,trace.jsonl,
progress.json,diagnostics.png. Full source hashes/hardware/software recorded.
Reused existing environment without package changes. Epoch-boundary resume only;
source/config mismatch is refused. Batch log includes raw InfoNCE,positive/negative
cosines,retrieval accuracy,CE/replay losses,gradient norms,feature norms/variance,
training accuracy and random-weight diagnostics.
Checkpoint captures model,optimizer,CPU reservoir and all RNG streams.

Final evaluation only after all50 epochs:10,000 clean and750,000 corrupted examples,
15 standard CIFAR10-C corruptions x5 severities. Both all10-class class-IL and
true-pair-masked task-IL, no augmentation at test, no test-based selection.
Compare with seed17 MSE and matched zero-loss runs as an initial diagnostic.
Further seeds are not automatically authorized by this first-run request.
