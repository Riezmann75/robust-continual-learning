import copy
import math
import unittest
import torch
from libs.training_schedule import cosine_for_epoch


class CosineTests(unittest.TestCase):
    def test_cycles_and_checkpoint_resume(self):
        for base_lr in (.012, .02):
            parameter = torch.nn.Parameter(torch.ones(1))
            opt = torch.optim.SGD([parameter], lr=base_lr, momentum=.9)
            for epoch in range(20):
                scheduler = cosine_for_epoch(opt, base_lr, 10, epoch)
                expected = base_lr * (1 + math.cos(math.pi*(epoch % 10)/10))/2
                self.assertAlmostEqual(opt.param_groups[0]['lr'], expected, places=12)
                if epoch == 4:
                    saved = copy.deepcopy(opt.state_dict())
                parameter.grad = torch.ones_like(parameter)
                opt.step()
                scheduler.step()
            restored = torch.optim.SGD([parameter], lr=base_lr, momentum=.9)
            restored.load_state_dict(saved)
            cosine_for_epoch(restored, base_lr, 10, 4)
            self.assertAlmostEqual(restored.param_groups[0]['lr'],
                                   base_lr*(1+math.cos(math.pi*.4))/2, places=12)
