import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import chain_aware_mixing as runner


class SeedTests(unittest.TestCase):
    def sample(self, seed):
        with patch.object(torch.cuda, 'is_available', return_value=True), patch.object(torch.cuda, 'manual_seed_all') as seed_all:
            runner.setup(seed, 'cuda')
            seed_all.assert_called_with(seed)
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)
        return (random.random(), np.random.random(), torch.rand(3),
                runner.MIXTURE_RNG.random(),
                runner.Reservoir(seed=seed + 1).rng.random())

    def test_repeatable_and_distinct_streams(self):
        for seed in (17, 42):
            a, b = self.sample(seed), self.sample(seed)
            self.assertEqual(a[:2], b[:2])
            torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)
            self.assertEqual(a[3:], b[3:])
        a, b = self.sample(17), self.sample(42)
        self.assertNotEqual(a[0], b[0])
        self.assertNotEqual(a[1], b[1])
        self.assertFalse(torch.equal(a[2], b[2]))
        self.assertNotEqual(a[3], b[3])
        self.assertNotEqual(a[4], b[4])
