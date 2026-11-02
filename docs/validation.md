# Validation

Final repository tests passed on2026-11-02 using CUDA on NVIDIA RTX3060Ti.
Python3.13.11,PyTorch2.10.0+cu128,torchvision0.25.0+cu128.
Command: python -m unittest discover -s tests -v
Result:7 tests passed (33.847 seconds).

Covers decoder shape/count/gradients;exact augmented CE and reconstruction targets;
replay and BN conventions;zero-augmented-loss baseline recovery;exact checkpoint
next-step reproduction;five-task interrupted/completed resume for both variants;
historical source hashes and evaluation sizes;explicit new configs.

Only short synthetic correctness training was performed for the new variants.
No full-budget augmented-CE or decoder experiment was launched.
