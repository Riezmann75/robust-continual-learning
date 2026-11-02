# Active schedule revision: 10 epochs per task

User changed the requested schedule on 2026-11-02 to 10 epochs per task (50 total).
The previous run was stopped after task1 epoch14 (4,382 updates,140,000 presentations);
its latest checkpoint, raw traces, original config and source snapshot remain in
experiments/metamix_cifar10/long_smallcnn_cuda. Partial next-epoch work is uncommitted.
It was not used as a parent checkpoint because no epoch10 checkpoint was retained.

The new run starts fresh at seed17 in experiments/metamix_cifar10/smallcnn_cuda_10epochs.
All other scientific settings in metamix_long.md remain unchanged.
All 50,000 clean training examples are retained; 500,000 presentations,
15,650 classifier updates (313 per epoch), and 46,950 mixer updates total.
Final evaluation remains the full clean test set and 15 corruptions x5 severities.
No test-based checkpoint selection. This remains a new first-order reconstruction,
not the Mac source or verified authors' implementation.

Launch/resume from repository root under WSL:
.venv-nlp/bin/python reproduce/metamix_long.py --device cuda --epochs-per-task 10 --output experiments/metamix_cifar10/smallcnn_cuda_10epochs

Detached launch metadata: experiments/metamix_cifar10/train_cuda_10epochs.process.json
Log: experiments/metamix_cifar10/train_cuda_10epochs.log
CUDA validation: experiments/metamix_cifar10/validation_10epochs/validation.json
The CLI default remains100 for compatibility; the new run explicitly supplies10.
The source/config mismatch guard remains in force.
