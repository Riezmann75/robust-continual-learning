# Completed historical results

The complete inventory now covers **22 completed runs**. See [all training settings](../docs/all_training_results.md) and [Class-IL/Task-IL corruption-category results](../docs/all_settings_corruptions.md). The historical summary below is retained for context.

Each run contains full per-corruption/severity evaluation.json, selected original
configuration fields in provenance.json, and exact hashed training-source snapshots.
These are historical results, not outputs from the new augmented-CE/decoder experiments.
The snapshots are retained rather than rewriting historical provenance to fit
the current CLI. Data,model checkpoints,raw training traces and machine-specific
configuration paths are excluded from Git. They remain in the original workspace.

All listed runs use10 epochs per task,all 50,000 CIFAR10 training examples,
and final evaluation on10,000 clean and 750,000 corrupted test examples.
Class-IL:all 10 outputs. Task-IL:known true class-pair mask.
The MetaMix comparison changes more than the objective (BN,mixing,normalization).
Feature MSE versus its lambda0 baseline has matched seed17/18/19 runs.
Other variants are seed17 diagnostics. Hyperparameter experiments are exploratory;
do not describe their best observed test score as an unbiased selected-model result.

Print the table: python scripts/summarize_results.py
