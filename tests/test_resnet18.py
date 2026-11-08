"""Scratch initialization and real continual-learning updates with ResNet18."""
import sys
import copy
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torchvision.models import resnet18
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import augce_cifar10 as augce
from libs.models import SmallCNN, Controller, ChainAwareController, backbone_metadata

class ResNet18Tests(unittest.TestCase):
    def test_scratch_shapes_and_metadata_rng(self):
        torch.set_num_threads(2)
        with patch('torchvision.models.resnet18', wraps=resnet18) as factory:
            model = SmallCNN(backbone='resnet18')
        factory.assert_called_once_with(weights=None)
        logits, features = model(torch.rand(2, 3, 32, 32))
        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(features.shape, (2, 512))
        state = torch.get_rng_state().clone()
        metadata = backbone_metadata('resnet18')
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(metadata['parameters'], sum(p.numel() for p in model.parameters()))
        self.assertFalse(metadata['pretrained'])

    def test_native_features_feed_controllers(self):
        context = torch.randn(2, 1026)
        state = (torch.zeros(2, 5), torch.zeros(2, 5))
        for controller in (Controller(512), ChainAwareController(512)):
            args = (torch.randn(2, 2, 3, 522),) if isinstance(controller, ChainAwareController) else ()
            m, w, _ = controller(context, state, *args)
            self.assertEqual(w.shape[-1], 3)
            (m.sum() + w[..., 0].sum()).backward()
            self.assertTrue(all(p.grad is not None for p in controller.parameters()))

    def test_bilevel_update_with_native_features(self):
        import matched_mixing as mixing
        import chain_aware_mixing as chain
        torch.set_num_threads(2)
        for module in (mixing, chain):
            torch.manual_seed(17)
            np.random.seed(17)
            module.MIXTURE_RNG = np.random.default_rng(170501)
            model = module.SmallCNN(backbone='resnet18')
            controller = module.Controller(feature_dimension=512)
            optimizer = torch.optim.SGD(model.parameters(), lr=.03)
            mixer_optimizer = torch.optim.SGD(controller.parameters(), lr=.05)
            raw = np.random.default_rng(7).integers(0,256,(2,32,32,3),dtype=np.uint8)
            _, row = module.step(model, controller, optimizer, mixer_optimizer,
                                 module.Reservoir(), module.initial_state('cpu'),
                                 raw, np.array([0,1]), 'cpu', j=1)
            self.assertTrue(np.isfinite(row['classifier_loss']))
            self.assertEqual(len(row['mixer_updates']), 1)

    def test_decoder_and_replay_update_encoder(self):
        augce.setup(17, 'cpu')
        model = augce.Network(decoder=True, backbone='resnet18')
        optimizer = torch.optim.SGD(model.parameters(), lr=.03)
        memory, streams = augce.Reservoir(), augce.Streams(17)
        raw = np.random.default_rng(7).integers(0, 256, (2,32,32,3), dtype=np.uint8)
        labels = np.array([0,1])
        before = model.net[0].weight.detach().clone()
        augce.step(model, optimizer, memory, streams, raw, labels, 'cpu')
        row = augce.step(model, optimizer, memory, streams, raw, labels, 'cpu')
        self.assertGreater(row['replay_ce'], 0)
        self.assertGreater(row['reconstruction_loss'], 0)
        self.assertFalse(torch.equal(before, model.net[0].weight))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        saved = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                   memory=memory.state_dict(), rng=augce.rng_state(streams, 'cpu')))
        first = augce.step(model, optimizer, memory, streams, raw, labels, 'cpu')
        expected = copy.deepcopy(model.state_dict())
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        memory.load_state_dict(saved['memory'])
        augce.restore_rng(saved['rng'], streams, 'cpu')
        second = augce.step(model, optimizer, memory, streams, raw, labels, 'cpu')
        self.assertEqual({k:v for k,v in first.items() if k != 'step_seconds'},
                         {k:v for k,v in second.items() if k != 'step_seconds'})
        self.assertTrue(all(torch.equal(v, expected[k]) for k,v in model.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
