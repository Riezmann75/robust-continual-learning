# Augmented CE and corresponding-input reconstruction

The current CNN has 288,746 parameters:two3x3 bias-free conv/BN/ReLU layers each at
32,64,128 channels,average pooling after stages1/2,global pooling,linear10-class head.
Normalization is internal:mean(.4914,.4822,.4465),std(.247,.243,.261).
No projection head,InfoNCE,feature-consistency MSE,JS,teacher or learned mixer
is used in the new augmented-CE path. DER++'s stored-logit MSE remains.

Let B be current examples,R_ce and R_logit independently sampled replay batches,
and A=concat(B,R_ce). Generate two independent random AugMix views of every A entry.
All three views share one joint encoder/classifier forward,including BN statistics.

L = CE(z0_B,y_B)
  +0.5 CE(z0_Rce,y_Rce)
  +0.2 mean_MSE(z_Rlogit,stored_logits)
  +lambda_aug *0.5[CE(z1_A,y_A)+CE(z2_A,y_A)]
  +lambda_recon *mean_MSE(D(h_all),x_all).

lambda_aug=1 by default. The two augmented losses are averaged,not each given
weight1 and then summed. Within each view,all current+replay examples are weighted
equally. Current-original CE and replay-original CE retain their separate weights.

Reconstruction is optional. Without --decoder there are no decoder parameters and
the reconstruction coefficient is recorded as 0. With --decoder,lambda_recon defaults
to 1. This is a proposed initial value,not calibrated against CE.
Targets are the corresponding unnormalized[0,1] pixels:
h0->x0,h1->x1,h2->x2. Neither features nor reconstruction gradients are detached.
PIL chains and view generation have no autograd. Logits stored in replay are
detached pre-update current logits. Only current examples enter the reservoir.

Decoder architecture:
- Linear128->2048,ReLU,reshape128x4x4.
- ConvTranspose2d128->64,kernel4,stride2,padding1,ReLU (8x8).
- ConvTranspose2d64->32,same geometry,ReLU (16x16).
- ConvTranspose2d32->3,same geometry,Sigmoid (32x32).
Decoder parameters 429,667;total encoder/classifier/decoder718,413.
The encoder/classifier remains unchanged. All parameters use the same SGD optimizer.
No additional BN layers are introduced by the decoder.

A separate original replay-logit forward updates encoder BN again when replay exists.
The decoder runs only on the 3N joint features,not on a second augmented batch.
During final classification evaluation the decoder is not invoked.

Sampling and checkpoint conventions match the feature-MSE experiments:
independent shuffle,insertion,replay-CE,replay-logit,operation and mixing RNG streams.
Source/config mismatches refuse resume. Model state includes decoder weights when enabled.

Tests cover loss coefficients,own-view pixel targets,encoder+decoder gradients,
parameter counts,zero-augmented-loss recovery of the matched DER++ baseline,
BN counts,short batches,pre-update stored logits,exact CUDA checkpoint roundtrip,
and interrupted/completed five-task resume without duplicate traces.
No full-training results exist yet for these two new variants.

Comparison limits:reconstruction is not a guarantee of robust or class-discriminative
representations. Matching augmented pixels may preserve nuisance variation.
Any later study must compare fixed seeds/sampling and avoid selecting weights on CIFAR10-C.
