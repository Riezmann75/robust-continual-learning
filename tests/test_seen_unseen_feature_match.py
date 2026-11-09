"""CPU gradient contracts and end-to-end steps; CUDA selected when available."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reproduce'))
import chain_aware_mixing as m
from libs.loss import seen_unseen_feature_loss


class FeatureMatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        np.random.seed(17)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_formula_detaches_targets(self):
        mixed = torch.randn(2, 4, 8, requires_grad=True)
        target = torch.randn(2, 3, 4, 8, requires_grad=True)
        loss = seen_unseen_feature_loss(mixed, target)
        expected = torch.stack([(mixed[v] - target[v, k].detach()).square().mean()
                                for v in range(2) for k in range(3)]).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertGreater(mixed.grad.norm().item(), 0)
        self.assertIsNone(target.grad)
        with self.assertRaises(ValueError):
            seen_unseen_feature_loss(mixed, target[:, :, :2])

    def test_outer_formula_and_frozen_classifier(self):
        net = m.SmallCNN().to(self.device).eval()
        ctl = m.Controller().to(self.device)
        x = torch.rand(3, 3, 32, 32, device=self.device)
        branches = [torch.stack((x*.2, x.flip(-1), x*.8)),
                    torch.stack((x*.7, x.flip(-2), x*.1))]
        desc = m.candidate_descriptors(net, x, branches)
        context = torch.randn(3, 258, device=self.device)
        state = tuple(v[:3] for v in m.initial_state(self.device))
        params = {k: v.detach() for k, v in net.named_parameters()}
        buffers = m.clone_buffers(net)
        targets = torch.randn(2, 3, 3, 128, device=self.device, requires_grad=True)
        before = copy.deepcopy(net.state_dict())
        loss, penalty, mm, w, ce = m.outer_objective(
            net, ctl, context, state, x, branches, params, buffers,
            torch.zeros(3, dtype=torch.long, device=self.device), 'feature_match',
            candidates=desc, target_features=targets)
        _, h = m.forward(net, torch.cat(m.mixed_views(x, branches, mm, w)), params, buffers)
        torch.testing.assert_close(loss, seen_unseen_feature_loss(h.reshape(2, 3, 128), targets))
        torch.testing.assert_close(loss, penalty)
        self.assertFalse(ce.requires_grad)
        loss.backward()
        self.assertIsNone(targets.grad)
        self.assertTrue(all(p.grad is None for p in net.parameters()))
        self.assertGreater(sum(p.grad.square().sum().item() for p in ctl.parameters()
                               if p.grad is not None), 0)
        for key, value in before.items():
            torch.testing.assert_close(net.state_dict()[key], value, rtol=0, atol=0)

    def test_training_step_seen_only_outer_and_replay(self):
        net = m.SmallCNN(backbone=getattr(self, 'backbone', 'smallcnn')).to(self.device)
        ctl = m.Controller(feature_dimension=net.feature_dimension).to(self.device)
        opt = torch.optim.SGD(net.parameters(), lr=.03)
        mo = torch.optim.SGD(ctl.parameters(), lr=.05)
        memory = m.Reservoir()
        state = m.initial_state(self.device)
        raw = np.random.randint(0, 256, (4, 32, 32, 3), dtype=np.uint8)
        labels = np.array([0, 1, 0, 1])
        original_chains, original_outer = m.chains, m.outer_objective
        sampled, outer_calls = [], []
        def sample(*args):
            value = original_chains(*args)
            sampled.append(value)
            return value
        def outer(*args, **kwargs):
            self.assertIs(args[5], sampled[0])  # seen, never unseen
            targets = kwargs['target_features']
            self.assertFalse(targets.requires_grad)
            self.assertEqual(targets.shape[:2], (2, 3))
            if outer_calls:
                self.assertIs(targets, outer_calls[0])  # cached over J steps
            outer_calls.append(targets)
            return original_outer(*args, **kwargs)
        for n in (4, 2):  # empty replay, then populated replay with short current batch
            sampled.clear(); outer_calls.clear()
            before = copy.deepcopy(ctl.state_dict())
            with patch.object(m, 'chains', sample), patch.object(m, 'outer_objective', outer):
                state, row = m.step(net, ctl, opt, mo, memory, state, raw[:n], labels[:n],
                                    self.device, outer_kind='feature_match')
            self.assertEqual(len(outer_calls), 4)
            self.assertEqual(len(row['mixer_updates']), 3)
            self.assertEqual(row['m_after'], .5)
            self.assertTrue(np.isfinite(row['feature_match_after']))
            self.assertTrue(all(u['grad_norm'] > 0 for u in row['mixer_updates']))
            self.assertTrue(any(not torch.equal(v, before[k]) for k, v in ctl.state_dict().items()))
            self.assertTrue(all(p.grad is None for p in net.parameters()))
        self.assertEqual(memory.seen, 6)
        self.assertEqual(row['replay_batch'], 4)

    def test_resnet18_feature_matching_step(self):
        self.backbone = 'resnet18'
        self.test_training_step_seen_only_outer_and_replay()


if __name__ == '__main__':
    unittest.main()
