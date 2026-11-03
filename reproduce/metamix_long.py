"""First-order, batch-context MetaMix reconstructed from the user's handoff."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libs.trainer import BiLevelTrainer
from libs import library_source_hashes, snapshot_source
import argparse
import hashlib
import json
import os
import platform
import random
import time
import numpy as np
from PIL import Image
import PIL
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call
import torchvision
from torchvision.datasets import CIFAR10
import augmix_operations as ops
from download_cifar100_benchmark import CORRUPTIONS, verify_arrays
from report_metamix_long import report

from libs.models import SmallCNN

from libs.models import BatchController as Controller

class Reservoir:
    def __init__(self, capacity=500):
        self.capacity = capacity
        self.seen = 0
        self.size = 0
        self.x = torch.empty(capacity, 32, 32, 3, dtype=torch.uint8)
        self.y = torch.empty(capacity, dtype=torch.long)
        self.logits = torch.empty(capacity, 10)
        self.rng = np.random.default_rng(18)

    def add(self, x, y, logits):
        for image, label, logit in zip(x, y, logits.detach().cpu()):
            slot = self.seen if self.seen < self.capacity else int(self.rng.integers(self.seen+1))
            self.seen += 1
            if slot < self.capacity:
                self.x[slot].copy_(torch.as_tensor(np.array(image, copy=True)))
                self.y[slot] = int(label)
                self.logits[slot].copy_(logit)
            self.size = min(self.seen, self.capacity)

    def sample(self, batch, device):
        idx = self.rng.choice(self.size, min(batch, self.size), replace=False)
        return tensor(self.x[idx].numpy(), device), self.y[idx].to(device), self.logits[idx].to(device)

def tensor(raw, device):
    return torch.as_tensor(np.array(raw, copy=True), device=device).permute(0, 3, 1, 2).float()/255

def norm(x):
    mean = x.new_tensor((.4914, .4822, .4465))[None, :, None, None]
    std = x.new_tensor((.2470, .2435, .2616))[None, :, None, None]
    return (x-mean)/std

from libs.loss import js

def chains(raw, pool, device):
    views = []
    for _ in range(2):
        branches = []
        for _ in range(3):
            batch = []
            for image in raw:
                im = Image.fromarray(image)
                for _ in range(int(np.random.randint(1, 4))):
                    im = pool[int(np.random.randint(len(pool)))](im, 3)
                batch.append(np.array(im, copy=True))
            branches.append(tensor(np.stack(batch), device))
        views.append(torch.stack(branches))
    return views

def blend(x, branch, m, w):
    return m*x + (1-m)*(branch*w[:, None, None, None, None]).sum(0)

def clone_buffers(model):
    return {k: v.detach().clone() for k, v in model.named_buffers()}

def forward(model, x, params=None, buffers=None):
    if params is None:
        return model(norm(x))
    return functional_call(model, (params, buffers), (norm(x),))

def context_for(model, x, y, replay):
    # Cloned buffers prevent the auxiliary context passes from committing BN updates.
    params = dict(model.named_parameters())
    model.train()
    z, f = forward(model, x, params, clone_buffers(model))
    ce = F.cross_entropy(z, y)
    if replay is None:
        dot, rce, rf = ce.new_zeros(()), ce.new_zeros(()), f.new_zeros(128)
    else:
        rx, ry, _ = replay
        rz, rfeatures = forward(model, rx, params, clone_buffers(model))
        rce = F.cross_entropy(rz, ry)
        grads = torch.autograd.grad(ce, tuple(params.values()))
        rgrads = torch.autograd.grad(rce, tuple(params.values()))
        dot = sum((g*r).sum() for g, r in zip(grads, rgrads))
        rf = rfeatures.detach().mean(0)
    return torch.cat((dot.detach().reshape(1), rce.detach().reshape(1),
                      f.detach().mean(0), rf)).unsqueeze(0)

def step(model, controller, optimizer, mixer_optimizer, memory, state, raw, labels, device, j=3):
    return BiLevelTrainer.update_metamix_long(model, controller, optimizer, mixer_optimizer, memory, state, raw, labels, device, j, blend=blend, chains=chains, clone_buffers=clone_buffers, context_for=context_for, forward=forward, js=js, ops=ops, tensor=tensor)

def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)

def source_hashes():
    base = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(base.glob('*.py'))} | library_source_hashes()

def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())

def restore_rng(rng):
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state_all(rng['cuda'])

def save(path, payload):
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)

def setup(seed, device):
    if device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This run requires an available CUDA device; no fallback')
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def evaluate(model, root, output, device, synthetic=None):
    model.eval()
    @torch.no_grad()
    def accuracy(raw, labels):
        correct = task_correct = 0
        for start in range(0, len(raw), 256):
            x = tensor(raw[start:start+256], device)
            y = torch.as_tensor(np.array(labels[start:start+256]), device=device)
            logits = forward(model, x)[0]
            correct += (logits.argmax(1) == y).sum().item()
            mask = torch.arange(10, device=device)[None, :]//2 == y[:, None]//2
            task_correct += (logits.masked_fill(~mask, -torch.inf).argmax(1) == y).sum().item()
        return dict(n=len(raw), class_il=correct/len(raw), task_il=task_correct/len(raw))
    if synthetic is not None:
        result = dict(validation_only=True, clean=accuracy(*synthetic))
    else:
        test = CIFAR10(root, train=False, download=False)
        target = root/'CIFAR-10-C'
        assert (target/'verified.json').exists()
        verify_arrays(target, test.targets)
        result = dict(clean=accuracy(test.data, test.targets), corruptions={})
        for name in CORRUPTIONS:
            arr = np.load(target/(name+'.npy'), mmap_mode='r')
            result['corruptions'][name] = {
                str(severity): accuracy(arr[(severity-1)*10000:severity*10000], test.targets)
                for severity in range(1, 6)}
            atomic_json(output/'evaluation.partial.json', result)
            print('Evaluated', name, flush=True)
        values = [v for sev in result['corruptions'].values() for v in sev.values()]
        result['corrupted_mean'] = {k: float(np.mean([v[k] for v in values]))
                                    for k in ('class_il', 'task_il')}
        result['corrupted_examples'] = sum(v['n'] for v in values)
    atomic_json(output/'evaluation.json', result)
    return result

def smoke(device, output):
    setup(17, device)
    model, controller = SmallCNN().to(device), Controller().to(device)
    assert sum(p.numel() for p in model.parameters()) == 288746
    raw = np.random.randint(0, 256, (4, 32, 32, 3), dtype=np.uint8)
    x = tensor(raw, device)
    logits, features = forward(model, x)
    assert logits.shape == (4, 10) and features.shape == (4, 128)
    assert abs(js(logits, logits, logits).item()) < 1e-6
    branches = torch.rand(3, *x.shape, device=device)
    assert torch.equal(blend(x, branches, x.new_tensor(1.), x.new_ones(3)/3), x)
    state = (torch.randn(1, 5, device=device), torch.randn(1, 5, device=device))
    m, w, _ = controller(torch.zeros(1, 258, device=device), state)
    assert m.item() == .5 and torch.allclose(w, torch.ones_like(w)/3)
    opt = torch.optim.SGD(model.parameters(), lr=.03)
    mixopt = torch.optim.SGD(controller.parameters(), lr=.05)
    memory = Reservoir()
    rows = []
    for _ in range(2):
        state, row = step(model, controller, opt, mixopt, memory, state, raw, np.array([0, 1, 0, 1]), device)
        rows.append(row)
    assert rows[0]['replay_ce'] == 0 and rows[1]['replay_ce'] > 0
    assert all(r['mixer_updates'][0]['grad_norm'] > 0 for r in rows)
    assert all(p.grad is None for p in model.parameters())
    assert memory.seen == 8 and memory.x.device.type == 'cpu'
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'smoke.json', dict(device=torch.cuda.get_device_name(), parameters=288746,
        logits_shape=list(logits.shape), features_shape=list(features.shape), rows=rows,
        passed=True, source_hashes=source_hashes()))
    print('CUDA smoke passed: shapes, parameter count, identity blend, JS, finite gradients with empty/populated replay', flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=['cuda'], default='cuda')
    p.add_argument('--root', type=Path, default=Path('experiments/metamix_cifar10/data'))
    p.add_argument('--output', type=Path, default=Path('experiments/metamix_cifar10/long_smallcnn_cuda'))
    p.add_argument('--epochs-per-task', type=int, default=100)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--pipeline-test', action='store_true')
    a = p.parse_args()
    if a.epochs_per_task < 1:
        p.error('--epochs-per-task must be positive')
    if a.smoke:
        smoke(a.device, a.output)
        return
    setup(17, a.device)
    a.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = a.output/'MetaMix'/'checkpoint.pt'
    checkpoint_path.parent.mkdir(exist_ok=True)
    config = dict(method='MetaMix', implementation='new reconstruction; first-order batch-context',
        seed=17, device=a.device, gpu=torch.cuda.get_device_name(), epochs_per_task=1 if a.pipeline_test else a.epochs_per_task,
        images_per_class=4 if a.pipeline_test else 5000, tasks=[[0,1],[2,3],[4,5],[6,7],[8,9]],
        batch=32, replay_batch=32, reservoir_capacity=500, J=3, classifier_lr=.03, mixer_lr=.05,
        alpha=.2, beta=.5, js_coefficient=1., channels=[32,64,128], parameters=288746,
        resolution=32, hidden_size=5, context_size=258, severity=3, views=2, chains=3, depth=[1,3],
        normalization_mean=[.4914,.4822,.4465], normalization_std=[.2470,.2435,.2616],
        pipeline_test=a.pipeline_test, source_hashes=source_hashes(), python=platform.python_version(),
        torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__,
        pillow=PIL.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        threads=4, deterministic_cudnn=True, tf32=False, root=str(a.root.resolve()))
    config_path = a.output/'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError('Config/source mismatch: use a fresh output directory')
    else:
        atomic_json(config_path, config)
    if a.pipeline_test:
        raw = np.random.randint(0, 256, (40,32,32,3), dtype=np.uint8)
        labels = np.repeat(np.arange(10), 4)
    else:
        if not (a.root/'CIFAR-10-C'/'verified.json').exists():
            raise RuntimeError('Complete dataset verification before full training')
        data = CIFAR10(a.root, train=True, download=False)
        raw, labels = data.data, np.array(data.targets)
        assert len(raw) == 50000 and all((labels == c).sum() == 5000 for c in range(10))
    indices = [np.flatnonzero(np.isin(labels, pair)).tolist() for pair in config['tasks']]
    index_path = a.output/'train_indices.json'
    if index_path.exists():
        assert json.loads(index_path.read_text()) == indices
    else:
        atomic_json(index_path, indices)
    model, controller = SmallCNN().to(a.device), Controller().to(a.device)
    assert sum(p.numel() for p in model.parameters()) == 288746
    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    mixer_optimizer = torch.optim.SGD(controller.parameters(), lr=.05)
    memory = Reservoir()
    state = (torch.randn(1, 5, device=a.device), torch.randn(1, 5, device=a.device))
    next_task = next_epoch = updates = 0
    trace_path = a.output/'trace.jsonl'
    if checkpoint_path.exists():
        # Only this trusted local checkpoint; CPU loading keeps reservoir and RNG tensors on CPU.
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert ckpt['config'] == config
        model.load_state_dict(ckpt['model'])
        controller.load_state_dict(ckpt['controller'])
        optimizer.load_state_dict(ckpt['optimizer'])
        mixer_optimizer.load_state_dict(ckpt['mixer_optimizer'])
        memory = ckpt['memory']
        state = tuple(s.to(a.device) for s in ckpt['state'])
        next_task, next_epoch, updates = ckpt['next_task'], ckpt['next_epoch'], ckpt['updates']
        restore_rng(ckpt['rng'])
    # Epoch-boundary durability: drop rows from an uncommitted partial epoch.
    if trace_path.exists():
        temp = trace_path.with_suffix('.tmp')
        with trace_path.open() as src, temp.open('w') as dst:
            for line in src:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row['update'] <= updates:
                    dst.write(json.dumps(row)+'\n')
        os.replace(temp, trace_path)
    epochs = config['epochs_per_task']
    print(json.dumps(dict(event='start_or_resume', pid=os.getpid(), next_task=next_task,
        next_epoch=next_epoch, updates=updates, config=config)), flush=True)
    atomic_json(a.output/'process.json', dict(pid=os.getpid(), host=platform.node(),
        started=time.time(), command=' '.join(__import__('sys').argv)))
    trainer = BiLevelTrainer(model, controller, optimizer, mixer_optimizer, memory, state, a.device, update=step)
    for task in range(next_task, 5):
        first = next_epoch if task == next_task else 0
        for epoch in range(first, epochs):
            start = time.monotonic()
            order = np.random.permutation(indices[task])
            with trace_path.open('a', buffering=1) as trace:
                for batch, pos in enumerate(range(0, len(order), 32)):
                    ix = order[pos:pos+32]
                    row = trainer.step(raw[ix], labels[ix], j=3)
                    state = trainer.state
                    updates += 1
                    row.update(task=task+1, epoch=epoch+1, global_epoch=task*epochs+epoch+1,
                               batch=batch, update=updates, time=time.time())
                    trace.write(json.dumps(row)+'\n')
                trace.flush()
                os.fsync(trace.fileno())
            nt, ne = (task+1, 0) if epoch+1 == epochs else (task, epoch+1)
            save(checkpoint_path, dict(config=config, model=model.state_dict(),
                controller=controller.state_dict(), optimizer=optimizer.state_dict(),
                mixer_optimizer=mixer_optimizer.state_dict(), memory=memory,
                state=tuple(s.cpu() for s in state), rng=rng_state(),
                next_task=nt, next_epoch=ne, updates=updates))
            summary = report(a.output, updates)
            print(json.dumps(dict(event='epoch', task=task+1, epoch=epoch+1, updates=updates,
                seconds=time.monotonic()-start, m=row['m_after'], mean_m=summary['epochs'][-1]['m_mean'],
                presentations=memory.seen)), flush=True)
    if not a.pipeline_test:
        assert updates == 5*epochs*313 and memory.seen == 5*epochs*10000
    report(a.output, updates)
    if not (a.output/'evaluation.json').exists():
        evaluate(model, a.root, a.output, a.device, (raw, labels) if a.pipeline_test else None)
    atomic_json(a.output/'complete.json', dict(updates=updates, presentations=memory.seen,
        completed=time.time(), pipeline_test=a.pipeline_test))
    print('Completed training and evaluation', flush=True)

if __name__ == '__main__':
    main()
