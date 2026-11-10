import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import chain_aware_mixing as m


class ChainSeverityTests(unittest.TestCase):
    def test_one_level_per_image_view_and_branch(self):
        images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
        levels = np.linspace(.1, 2.9, 12)
        observed = []
        def record(im, level, *, level_is_sampled):
            self.assertTrue(level_is_sampled)
            observed.append(level)
            return im
        def randint(*args):
            return 3 if args == (1, 4) else 0
        with patch.object(m.ops, 'sample_level', side_effect=levels), patch.object(np.random, 'randint', side_effect=randint):
            m.chains(images, [record], 'cpu', chain_severity=True)
        np.testing.assert_equal(observed, np.repeat(levels, 3))

    def test_operators_use_supplied_level_without_resampling(self):
        im = Image.new('RGB', (32, 32))
        with patch.object(m.ops, 'sample_level', side_effect=AssertionError('unexpected redraw')):
            for op in m.ops.augmentations:
                op(im, 2.4, level_is_sampled=True)
        with patch.object(m.ops, 'sample_level', return_value=2.4) as sample:
            m.ops.rotate(im, 3)
            sample.assert_called_once_with(3)
