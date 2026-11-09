import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import virtual_mixer as vm


class VirtualMixerTests(unittest.TestCase):
    def test_hypergradient_matches_finite_difference(self):
        torch.manual_seed(4)
        net = vm.base.SmallCNN().double().eval()
        torch.set_num_threads(2)
        x, u = torch.rand(2, 3, 32, 32, dtype=torch.double), torch.rand(2, 3, 32, 32, dtype=torch.double)
        labels = torch.tensor([0, 1])
        def objective(weight):
            inner = F.cross_entropy(vm.base.forward(net, x + weight * u)[0], labels)
            clean, unseen, _ = vm.meta_loss(net, inner, .1, x, u, labels)
            return clean + unseen
        weight = torch.tensor(.4, dtype=torch.double, requires_grad=True)
        exact, = torch.autograd.grad(objective(weight), weight)
        eps = 1e-5
        numeric = (objective(weight.detach()+eps) - objective(weight.detach()-eps))/(2*eps)
        self.assertGreater(exact.abs().item(), 1e-10)
        torch.testing.assert_close(exact, numeric, atol=1e-8, rtol=.01)

    def test_full_step_with_and_without_replay(self):
        torch.manual_seed(17)
        np.random.seed(17)
        torch.set_num_threads(2)
        labels = np.repeat(np.arange(2), 20)
        support, query = vm.split_task_indices(labels, [[0, 1]])
        self.assertFalse(set(support[0]) & set(query[0]))
        raw = np.random.randint(0, 256, (40,32,32,3), dtype=np.uint8)
        net, ctl = vm.base.SmallCNN(), vm.base.Controller()
        opt, mo = torch.optim.SGD(net.parameters(), lr=.03), torch.optim.SGD(ctl.parameters(), lr=.05)
        state, memory = vm.base.initial_state('cpu'), vm.base.Reservoir()
        for ix in (support[0][::9], support[0][1::13]):
            before = [p.detach().clone() for p in ctl.parameters()]
            state, row = vm.step(net, ctl, opt, mo, memory, state, raw[ix], labels[ix], 'cpu',
                                 query_batch=(raw[query[0]], labels[query[0]]), kind='infonce', temperature=.1)
            self.assertGreater(row['meta_grad_norm'], 0)
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,ctl.parameters())))
            self.assertAlmostEqual(row['mixer_loss'], row['query_clean_ce']+row['query_unseen_ce'], places=5)
