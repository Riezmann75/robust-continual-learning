"""Shared trainer state and independently runnable source snapshots."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'reproduce'))
import augce_cifar10 as single
import matched_mixing as bilevel
from libs import library_source_hashes, snapshot_source
from libs.trainer import BiLevelTrainer, SingleLevelTrainer


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.random.default_rng(4).integers(0, 256, (2, 32, 32, 3), dtype=np.uint8)
        self.labels = np.array([0, 1])

    def test_single_level_retains_resources_and_replay(self):
        single.setup(17, 'cpu')
        model, memory, streams = single.Network(), single.Reservoir(), single.Streams(17)
        optimizer = torch.optim.SGD(model.parameters(), lr=.03)
        trainer = SingleLevelTrainer(model, optimizer, memory, streams, 'cpu', update=single.step)
        before = model.head.weight.detach().clone()
        first = trainer.step(self.raw, self.labels)
        second = trainer.step(self.raw, self.labels)
        self.assertEqual(first['replay_batch'], 0)
        self.assertEqual(second['replay_batch'], 2)
        self.assertEqual(memory.seen, 4)
        self.assertFalse(torch.equal(before, model.head.weight))

    def test_bilevel_advances_and_restores_state(self):
        # Exercise the library on CPU; the historical CLI setup requires CUDA.
        torch.set_num_threads(2)
        torch.manual_seed(17)
        np.random.seed(17)
        model, controller = bilevel.SmallCNN(), bilevel.Controller()
        memory = bilevel.Reservoir()
        state = bilevel.initial_state('cpu')
        trainer = BiLevelTrainer(model, controller,
                                torch.optim.SGD(model.parameters(), lr=.03),
                                torch.optim.SGD(controller.parameters(), lr=.05),
                                memory, state, 'cpu', update=bilevel.step, j=1)
        row, debug = trainer.step(self.raw, self.labels, debug=True)
        self.assertEqual(len(row['mixer_updates']), 1)
        self.assertFalse(debug['stored'].requires_grad)
        self.assertFalse(torch.equal(state[0][:2], trainer.state[0][:2]))
        self.assertTrue(torch.equal(state[0][2:], trainer.state[0][2:]))
        self.assertTrue(all(not value.requires_grad for value in trainer.state))
        trainer.step(self.raw, self.labels)
        self.assertEqual(memory.seen, 4)

    def test_snapshot_contains_hashed_library_and_imports_outside_repo(self):
        hashes = single.source_hashes()
        self.assertTrue(set(library_source_hashes()).issubset(hashes))
        with tempfile.TemporaryDirectory() as temp:
            snapshot = Path(temp) / 'source'
            for name, expected in hashes.items():
                snapshot_source(ROOT / 'reproduce', snapshot, name)
                self.assertEqual(hashlib.sha256((snapshot / name).read_bytes()).hexdigest(), expected)
            result = subprocess.run([sys.executable, str(snapshot / 'augce_cifar10.py'), '--help'],
                                    cwd=temp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
