"""Classifier, reconstruction, and mixing-controller architectures."""
import torch
from torch import nn
from torch.nn import functional as F


class SmallCNN(nn.Module):
    """Shared 128-feature classifier; inputs must already be normalized."""

    def __init__(self):
        super().__init__()
        layers = []
        prev = 3
        for stage, width in enumerate((32, 64, 128)):
            for _ in range(2):
                layers += [nn.Conv2d(prev, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU()]
                prev = width
            if stage < 2:
                layers.append(nn.AvgPool2d(2))
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        f = self.net(x).mean((2, 3))
        return (self.head(f), f)


class ImageDecoder(nn.Module):
    """Shared decoder: raw pooled128 feature -> RGB32x32 in [0,1]."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128, 128 * 4 * 4), nn.ReLU(), nn.Unflatten(1, (128, 4, 4)), nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(), nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(), nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid())

    def forward(self, features):
        return self.net(features)


class ReconstructionNetwork(SmallCNN):
    """Pixel-input classifier with an optional shared RGB decoder."""

    def __init__(self, decoder=False):
        super().__init__()
        self.register_buffer('pixel_mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1))
        self.decoder = ImageDecoder() if decoder else None

    def forward(self, x, features=False):
        logits, pooled = super().forward((x - self.pixel_mean) / self.pixel_std)
        return (logits, pooled) if features else logits


class Network(SmallCNN):
    """Pixel-input classifier without a reconstruction branch."""

    def __init__(self):
        super().__init__()
        self.register_buffer('pixel_mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1))

    def forward(self, x, features=False):
        logits, pooled = super().forward((x - self.pixel_mean) / self.pixel_std)
        return (logits, pooled) if features else logits


class ChainAwareController(nn.Module):
    """Score each candidate chain with fixed original-image weight 0.5."""

    def __init__(self):
        super().__init__()
        self.cell = nn.LSTMCell(258, 5)
        self.scorer = nn.Sequential(nn.Linear(5 + 128 + 10, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, context, state, candidates):
        state = self.cell(context, state)
        assert candidates.shape == (2, len(context), 3, 138)
        hidden = state[0][None, :, None, :].expand(2, -1, 3, -1)
        scores = self.scorer(torch.cat((hidden, candidates), dim=-1)).squeeze(-1)
        w = scores.softmax(-1)
        m = scores.new_full((2, len(context)), 0.5)
        return (m, w, state)


class InfoNCEReconstructionNetwork(SmallCNN):
    """Initialize the decoder without advancing the classifier RNG stream."""

    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.decoder = ImageDecoder()
        self.register_buffer('pixel_mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1))

    def forward(self, x, features=False):
        logits, pooled = super().forward((x - self.pixel_mean) / self.pixel_std)
        return (logits, pooled) if features else logits


class Controller(nn.Module):
    """Learn original-image and branch weights for each recurrent batch slot."""

    def __init__(self):
        super().__init__()
        self.cell = nn.LSTMCell(258, 5)
        self.head = nn.Linear(5, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, context, state):
        state = self.cell(context, state)
        out = self.head(state[0])
        return (out[:, 3].sigmoid(), out[:, :3].softmax(1), state)


class BatchController(nn.Module):
    """Historical controller producing one set of weights for the whole batch."""

    def __init__(self):
        super().__init__()
        self.cell = nn.LSTMCell(258, 5)
        self.head = nn.Linear(5, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, context, state):
        state = self.cell(context, state)
        out = self.head(state[0])[0]
        return (out[3].sigmoid(), out[:3].softmax(0), state)


class MultiHeadNetwork(SmallCNN):
    """Four independent feature branches and classification heads."""

    def __init__(self):
        super().__init__()
        del self.head
        self.branches = nn.ModuleList([nn.Sequential(nn.Linear(128, 32), nn.ReLU()) for _ in range(4)])
        self.head = nn.ModuleList([nn.Linear(32, 10) for _ in range(4)])
        self.register_buffer('pixel_mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1))

    def forward(self, x, features=False):
        pooled = self.net((x - self.pixel_mean) / self.pixel_std).mean((2, 3))
        h = torch.stack([branch(pooled) for branch in self.branches], dim=1)
        z = torch.stack([head(h[:, k]) for k, head in enumerate(self.head)], dim=1)
        return (z, h) if features else z
