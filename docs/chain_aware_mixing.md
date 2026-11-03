# Chain-aware fixed-m mixing experiment

New method, not an author-code reproduction of MetaMix. Start fresh; do not resume older mixers.

The classifier objective is clean current CE + 0.5 replay CE + 0.2 replay-logit MSE + consistency on original and two pseudo-seen mixed views. Classifier consistency is either class-aware InfoNCE (the default, weight 1, temperature 0.1) or JS (weight 1, `--consistency js`). The outer mixer objective remains pseudo-unseen three-view JS, weight 1, with a fixed updated classifier. Outer clean CE is reported but has zero mixer gradient. There is no hypergradient through the classifier update and no explicit post-update replay-retention objective.

For each batch, independently sample two views of three chains from each disjoint pseudo-seen/unseen pool, as in the previous matched runner. Each candidate descriptor concatenates its raw pooled feature minus original feature (128 coordinates) and softmax probabilities minus original probabilities (10 coordinates). Descriptors are detached. Seen descriptors use the classifier before its update; unseen descriptors use it afterward and remain fixed during J=3 mixer steps.

The LSTM retains the existing detached 258-dimensional clean current/replay context and hidden size 5. A shared Linear(143,32)-ReLU-Linear(32,1) scores the concatenation of its hidden state and each 138-dimensional candidate descriptor. The output layer starts at zero, giving uniform weights. Scores are softmaxed across three chains separately per example and view. Chain permutations permute weights; no operation identity features or raw clean-feature input are added to the scorer. The original clean/replay features remain in the LSTM context. m is exactly 0.5 and has no learnable parameter.

Classifier original and two mixed views use one joint training-mode forward, followed by independent logit-replay forward. Each descriptor extraction uses one joint original-plus-six-candidates eval forward, so it does not mutate BatchNorm statistics. All outer original/two mixed views also use a joint eval forward. LSTM state advances once using pre-solver parameters, preserving existing 64 batch-slot semantics.

Configuration: CUDA float32; seed 17; native 32x32 six-convolution CNN (288,746 parameters); all 50,000 clean training images; five sequential class pairs; 10 epochs/task; current/replay batch 32 including final short batch; reservoir 500; classifier SGD 0.03; mixer SGD 0.05, J=3. No crop/flip or test-time augmentation. Inherit previous matched-run normalization (0.2470,0.2435,0.2616 std), operation sampling and reservoir RNG conventions.

Final evaluation: all 10,000 clean test images and all 750,000 CIFAR-10-C images (15 corruptions, five severities); class-IL and true-pair task-IL. No test-driven checkpoint selection.

Epoch-boundary checkpoints preserve model/controller/optimizers/replay/RNG/recurrent state; source/config mismatches are refused. Logs record per-example/view weights before/after, fixed m, the selected classifier consistency, outer JS and mixer gradient norms. Reports include chain-weight entropy and maximum weight. Additional candidate forwards increase runtime. Compared with the earlier non-chain-aware mixer, this changes controller inputs, per-view weighting, m and outer objective together, so that comparison is not an isolated ablation. Within the chain-aware method, the new JS configuration isolates the classifier consistency objective.

Run from the standalone repository using the workspace Python:

```sh
../.venv-nlp/bin/python scripts/run_experiment.py configs/chain_aware_mixing.json --run
```

Run the JS classifier comparison in its own output directory:

```sh
python scripts/run_experiment.py configs/chain_aware_js.json --run
```

This configuration uses `experiments/chain_aware_js/seed17`. It changes only the
classifier consistency from InfoNCE to JS; seed 17, fixed m=0.5, controller,
CE + DER++, outer JS, and the 10-epoch/task budget stay the same. Temperature is
retained in the configuration for comparability but is unused by JS. This is a
new comparison, with no full-budget results reported yet. Use a fresh output
directory because source/configuration checks prevent cross-version resume.

Validation:

```sh
../.venv-nlp/bin/python -m unittest discover -s tests -p test_chain_aware.py -v
```
