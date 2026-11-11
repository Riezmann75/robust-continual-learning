"""Compare the actual solver's mixer gradient to a numerical derivative."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import unrolled_mixing as solver
from libs.models import ChainAwareController, Controller


class TinyClassifier(nn.Module):
    feature_dimension = 3

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 10)

    def forward(self, x):
        features = x.mean((2, 3))
        return self.head(features), features


class GradientTest(unittest.TestCase):
    def evaluate(self, variant, rate, shift=0., direction=None):
        torch.manual_seed(42)
        np.random.seed(42)
        base = solver.base
        model = TinyClassifier().double()
        ctor = Controller if variant == 'metamix' else ChainAwareController
        ctl = ctor(feature_dimension=3).double()
        if direction is not None:
            with torch.no_grad():
                for p, d in zip(ctl.parameters(), direction):
                    p.add_(shift * d)
        opt = torch.optim.SGD(model.parameters(), lr=rate)
        mo = torch.optim.SGD(ctl.parameters(), lr=.02)
        raw = np.random.default_rng(34).integers(256, size=(3,32,32,3), dtype=np.uint8)
        tensor = base.tensor
        with patch.object(base, 'MIXTURE_RNG', np.random.default_rng(7)), patch.object(
                base, 'tensor', lambda x, device: tensor(x, device).double()):
            _, row = solver.step(model, ctl, opt, mo, base.Reservoir(),
                                 tuple(v.double() for v in base.initial_state('cpu')),
                                 raw, [0,1,0], 'cpu', variant=variant, j=1,
                                 kind='js', outer_kind='js', temperature=.1, mixer='learned')
        entry = row['mixer_updates'][0]
        return entry['outer_ce'] + entry['js'], [p.grad for p in ctl.parameters()]

    def test_gradient_paths(self):
        for variant in ('metamix', 'chain_random'):
            with self.subTest(variant=variant):
                _, grad = self.evaluate(variant, .012)
                norm = sum(g.square().sum() for g in grad).sqrt().item()
                self.assertGreater(norm, 0)
                direction = [g / norm for g in grad]
                eps = 1e-4
                plus, _ = self.evaluate(variant, .012, eps, direction)
                minus, _ = self.evaluate(variant, .012, -eps, direction)
                self.assertAlmostEqual((plus-minus)/(2*eps), norm, delta=1e-9 + 1e-5*norm)
                _, direct = self.evaluate(variant, 0)
                self.assertEqual(any(g.abs().max() > 0 for g in direct), variant == 'metamix')
