"""Corrected unchanged controller: seen InfoNCE, unseen JS ablation."""
import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
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
from report_matched_mixing import report
import shutil

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        prev = 3
        for stage, width in enumerate((32, 64, 128)):
            for _ in range(2):
                layers += [nn.Conv2d(prev, width, 3, padding=1, bias=False),
                           nn.BatchNorm2d(width), nn.ReLU()]
                prev = width
            if stage < 2:
                layers.append(nn.AvgPool2d(2))
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        f = self.net(x).mean((2, 3))
        return self.head(f), f

class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.LSTMCell(258, 5)
        self.head = nn.Linear(5, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, context, state):
        state = self.cell(context, state)
        out = self.head(state[0])
        return out[:,3].sigmoid(), out[:,:3].softmax(1), state

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

    def sample(self, batch, device, raw=False):
        idx = self.rng.choice(self.size, min(batch, self.size), replace=False)
        images = self.x[idx].numpy()
        return (images if raw else tensor(images, device)), self.y[idx].to(device), self.logits[idx].to(device)

def tensor(raw, device):
    return torch.as_tensor(np.array(raw, copy=True), device=device).permute(0, 3, 1, 2).contiguous().float()/255

def norm(x):
    mean = x.new_tensor((.4914, .4822, .4465))[None, :, None, None]
    std = x.new_tensor((.2470, .2435, .2616))[None, :, None, None]
    return (x-mean)/std

def js(*logits):
    logps = [F.log_softmax(z, dim=1) for z in logits]
    ps = [p.exp() for p in logps]
    mean = torch.stack(ps).mean(0).clamp_min(1e-8).log()
    return sum((p*(lp-mean)).sum(1).mean() for p, lp in zip(ps, logps))/len(ps)

def contrastive_scores(h0,h1,h2,temperature=.1,labels=None):
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    n = len(h0)
    if n < 1 or h1.shape != h0.shape or h2.shape != h0.shape:
        raise ValueError('Expected three nonempty aligned feature matrices')
    u = F.normalize(torch.cat((h0,h1,h2)),p=2,dim=1,eps=1e-8)
    scores = u @ u.T / temperature
    ids = torch.arange(n,device=u.device).repeat(3)
    diagonal = torch.eye(3*n,dtype=torch.bool,device=u.device)
    positives = ids[:,None].eq(ids[None,:]) & ~diagonal
    if labels is None or labels.ndim != 1 or len(labels) != n:
        raise ValueError('One class label per original batch entry is required')
    classes = labels.to(u.device).repeat(3)
    different_class = classes[:,None].ne(classes[None,:])
    allowed = positives | different_class
    return scores.masked_fill(~allowed,-torch.inf), positives

def infonce_loss(h0,h1,h2,temperature=.1,labels=None):
    # All 3N anchors, two positives each. Average individual positive log probabilities.
    scores, positives = contrastive_scores(h0,h1,h2,temperature,labels)
    normalizer = torch.logsumexp(scores,dim=1)
    positive_score = scores[positives].view(len(scores),2).mean(1)
    return (normalizer-positive_score).mean()


def consistency_loss(logits,features,labels,kind,temperature):
    if kind=='js':return js(*logits)
    return infonce_loss(*features,temperature=temperature,labels=labels)

# Extra random mixture draws never consume the historical shuffle/operator RNG.
MIXTURE_RNG = np.random.default_rng(170501)

def random_weights(n,device):
    m=torch.tensor(MIXTURE_RNG.beta(1,1,(2,n)),dtype=torch.float32,device=device)
    w=torch.tensor(MIXTURE_RNG.dirichlet(np.ones(3),(2,n)),dtype=torch.float32,device=device)
    return m,w

def mixed_views(x,branches,m,w):
    return [blend(x,v,m[i] if m.ndim==2 else m,w[i] if w.ndim==3 else w) for i,v in enumerate(branches)]

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
    assert m.shape == (len(x),) and w.shape == (len(x),3)
    return m[:,None,None,None]*x + (1-m[:,None,None,None])*(branch*w.T[:,:,None,None,None]).sum(0)

def clone_buffers(model):
    return {k: v.detach().clone() for k, v in model.named_buffers()}

def forward(model, x, params=None, buffers=None):
    if params is None:
        return model(norm(x))
    return functional_call(model, (params, buffers), (norm(x),))

def context_for(model, x, y, replay):
    # Algorithm 3 batch-level gradient/loss statistics, with individual features.
    # Positional cyclic pairing is explicit: the paper does not specify alignment.
    params = dict(model.named_parameters())
    model.train()
    z, f = forward(model, x, params, clone_buffers(model))
    ce = F.cross_entropy(z, y)
    b = len(x)
    if replay is None:
        dot, rce = ce.new_zeros(()), ce.new_zeros(())
        er, et = torch.zeros_like(f), f.detach()
    else:
        rx, ry, _ = replay
        rz, rf = forward(model, rx, params, clone_buffers(model))
        rce = F.cross_entropy(rz, ry)
        grads = torch.autograd.grad(ce, tuple(params.values()))
        rgrads = torch.autograd.grad(rce, tuple(params.values()))
        dot = sum((g*r).sum() for g,r in zip(grads,rgrads))
        r = len(rx)
        er = torch.cat((rf.detach()[torch.arange(b,device=x.device)%r],rf.detach()))
        et = torch.cat((f.detach(),f.detach()[torch.arange(r,device=x.device)%b]))
    n = len(et)
    return torch.cat((dot.detach().expand(n,1),rce.detach().expand(n,1),er,et),dim=1)

def initial_state(device):
    # Separate fixed slots for current [0:32] and replay [32:64]. Not image IDs.
    return (torch.randn(64,5,device=device),torch.randn(64,5,device=device))

def active_slots(b,r,device):
    return torch.cat((torch.arange(b,device=device),32+torch.arange(r,device=device)))

def outer_objective(model,controller,context,incoming,x,branches,params,buffers,labels,kind='js',temperature=.1):
    assert not model.training
    om,ow,_=controller(context,incoming)
    joined=torch.cat((x,*mixed_views(x,branches,om,ow)))
    z,h=forward(model,joined,params,buffers)
    zs=list(z.chunk(3));hs=list(h.chunk(3))
    zs[0]=zs[0].detach();hs[0]=hs[0].detach()
    ce=F.cross_entropy(zs[0],labels)
    penalty=consistency_loss(zs,hs,labels,kind,temperature)
    return ce+penalty,penalty,om,ow,ce

def step(model,controller,optimizer,mixer_optimizer,memory,state,raw,labels,device,j=3,debug=False,mixer='learned',kind='infonce',temperature=.1,outer_kind='js'):
    x=tensor(raw,device);y=torch.as_tensor(labels,device=device)
    ce_raw=memory.sample(32,device,raw=True) if memory.size else None
    replay=(tensor(ce_raw[0],device),ce_raw[1],ce_raw[2]) if ce_raw is not None else None
    distill=memory.sample(32,device) if memory.size else None
    raw_a=np.concatenate((raw,ce_raw[0])) if ce_raw is not None else raw
    xa=tensor(raw_a,device);ya=torch.cat((y,ce_raw[1])) if ce_raw is not None else y
    b,r=len(raw),0 if ce_raw is None else len(ce_raw[0])
    slots=active_slots(b,r,device)
    context=context_for(model,x,y,replay) if mixer=='learned' else None
    incoming=tuple(v[slots].detach() for v in state)
    next_state=tuple(v.detach().clone() for v in state)
    with torch.no_grad():
        if mixer=='learned':
            m,w,advanced=controller(context,incoming)
            for target,value in zip(next_state,advanced):target[slots]=value
        else:m,w=random_weights(b+r,device)
    perm=np.random.permutation(9);q=int(np.random.randint(5,8))
    seen=chains(raw_a,[ops.augmentations[i] for i in perm[:q]],device)
    # Generate unseen even for random-mixer controls to preserve RNG consumption.
    unseen=chains(raw_a,[ops.augmentations[i] for i in perm[q:]],device)
    digest=hashlib.sha256(raw_a.tobytes()+ya.cpu().numpy().tobytes()+perm.tobytes())
    for v in seen+unseen:digest.update(v.cpu().numpy().tobytes())
    model.train();optimizer.zero_grad(set_to_none=True)
    joined=torch.cat((xa,*mixed_views(xa,seen,m,w)))
    z,h=forward(model,joined);zs=z.chunk(3);hs=h.chunk(3);logits=zs[0]
    old_logits=logits[:b].detach().clone()
    ce=F.cross_entropy(logits[:b],y)
    replay_ce=F.cross_entropy(logits[b:],replay[1]) if r else ce.new_zeros(())
    replay_mse=F.mse_loss(forward(model,distill[0])[0],distill[2]) if distill else ce.new_zeros(())
    penalty=consistency_loss(zs,hs,ya,kind,temperature)
    loss=ce+.5*replay_ce+.2*replay_mse+penalty
    loss.backward()
    if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite classifier loss/gradient')
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    model.eval();params={k:v.detach() for k,v in model.named_parameters()};buffers=clone_buffers(model)
    outer_rows=[]
    if mixer=='learned':
        for _ in range(j):
            mixer_optimizer.zero_grad(set_to_none=True)
            outer_loss,outer_penalty,om,ow,outer_ce=outer_objective(model,controller,context,incoming,xa,unseen,params,buffers,ya,outer_kind,temperature)
            outer_loss.backward()
            grads=[p.grad for p in controller.parameters() if p.grad is not None]
            if not torch.isfinite(outer_loss) or not grads or not all(torch.isfinite(g).all() for g in grads):raise RuntimeError('Nonfinite mixer loss/gradient')
            grad_norm=torch.stack([g.square().sum() for g in grads]).sum().sqrt().item()
            outer_rows.append(dict(m=om.mean().item(),m_per_example=om.detach().cpu().tolist(),w=ow.detach().cpu().tolist(),consistency_loss=outer_penalty.item(),grad_norm=grad_norm))
            mixer_optimizer.step()
        with torch.no_grad():
            _,after_penalty,after_m,after_w,_=outer_objective(model,controller,context,incoming,xa,unseen,params,buffers,ya,outer_kind,temperature)
    else:
        with torch.no_grad():
            uz,uh=forward(model,torch.cat((xa,*mixed_views(xa,unseen,m,w))),params,buffers)
            after_penalty=consistency_loss(uz.chunk(3),uh.chunk(3),ya,kind,temperature)
            outer_ce=F.cross_entropy(uz.chunk(3)[0],ya)
        after_m,after_w=m,w
    memory.add(raw,labels,old_logits)
    row=dict(m_before=m.mean().item(),m_after=after_m.mean().item(),
        m_before_per_example=m.flatten().cpu().tolist(),m_after_per_example=after_m.flatten().cpu().tolist(),
        w_before=w.cpu().tolist(),w_after=after_w.cpu().tolist(),mixer=mixer,consistency=kind,outer_consistency=outer_kind,
        sampling_sha256=digest.hexdigest(),seen_consistency=penalty.item(),outer_consistency_after=after_penalty.item(),
        current_ce=ce.item(),replay_ce=replay_ce.item(),replay_mse=replay_mse.item(),classifier_loss=loss.item(),
        outer_ce=outer_ce.item(),mixer_updates=outer_rows,q=q,batch_size=b,replay_batch=r,augmented_batch=b+r,
        memory_size=memory.size,presentations=memory.seen)
    if debug:return next_state,row,dict(stored=old_logits,m=m,w=w,context=context)
    return next_state,row

def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)

def source_hashes():
    base=Path(__file__).parent
    names=['matched_infonce_js.py','report_matched_mixing.py','augmix_operations.py','AUGMIX_LICENSE',
           'download_cifar100_benchmark.py','download_cifar10_benchmark.py']
    return {name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in names}

def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),mixture=MIXTURE_RNG.bit_generator.state)

def restore_rng(rng):
    MIXTURE_RNG.bit_generator.state=rng['mixture']
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
    global MIXTURE_RNG
    MIXTURE_RNG=np.random.default_rng(170501)
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mixer',choices=['learned'],default='learned')
    p.add_argument('--consistency',choices=['infonce'],default='infonce')
    p.add_argument('--temperature',type=float,default=.1)
    p.add_argument('--device', choices=['cuda'], default='cuda')
    p.add_argument('--root', type=Path, default=Path('experiments/metamix_cifar10/data'))
    p.add_argument('--output', type=Path, default=Path('experiments/matched_mixing/learned_infonce_js/seed17'))
    p.add_argument('--epochs-per-task', type=int, default=10)
    p.add_argument('--stop-after-epochs',type=int,default=0)
    p.add_argument('--pipeline-test', action='store_true')
    a = p.parse_args()
    if not np.isfinite(a.temperature) or a.temperature<=0:p.error('positive finite temperature required')
    if a.epochs_per_task < 1:
        p.error('--epochs-per-task must be positive')
    if a.stop_after_epochs and not a.pipeline_test:
        p.error('stop-after-epochs is validation only')
    setup(17, a.device)
    a.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = a.output/'MetaMix'/'checkpoint.pt'
    checkpoint_path.parent.mkdir(exist_ok=True)
    config = dict(method='Corrected learned mixer: seen InfoNCE + unseen JS + DER++',outer_consistency='js',outer_coefficient=1.,mixer=a.mixer,consistency=a.consistency,temperature=a.temperature,consistency_coefficient=1.,
        random_mixture='Beta(1,1),Dirichlet(1,1,1); independent image/view, stream170501',
        sampling='preserve legacy global shuffle/ops and reservoir RNG; isolate mixture draws; generate both seen/unseen in all variants',
        matching_limit='learned weights shared across views as current run; random weights independently sampled across views as requested', implementation='joint three-view forwards; replay augmentation; per-example slot state; pre-solver state advance',
        solver='Algorithm 2 explicit fixed-updated-classifier partial gradient; original CE contributes zero mixer gradient; no unrolled hypergradient',
        context_alignment='broadcast batch dot/replay CE; individual er,et cyclic positional pairing for unequal batch sizes; zero er when empty',
        recurrent_state='64 persistent batch slots: current0:32 replay32:64; inactive slots unchanged; detach between batches; save pre-solver transition',
        bn='joint clean+two seen views train forward then separate distillation; context cloned buffers; joint clean+two unseen outer eval forward',
        unresolved='paper lacks precise feature pairing and autograd details; first-order interpretation not verified author code',
        parent_run_sha256='b45bad8e80b06ab9579cf72f81df7e1c93055944c544adb3e5239a25534fd41e',
        seed=17, device=a.device, gpu=torch.cuda.get_device_name(), epochs_per_task=1 if a.pipeline_test else a.epochs_per_task,
        images_per_class=17 if a.pipeline_test else 5000, tasks=[[0,1],[2,3],[4,5],[6,7],[8,9]],
        batch=32, replay_batch=32, reservoir_capacity=500, J=3 if a.mixer=='learned' else 0, classifier_lr=.03, mixer_lr=.05,
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
        snapshot=a.output/'source';snapshot.mkdir(exist_ok=True)
        for name in config['source_hashes']:
            shutil.copy2(Path(__file__).parent/name,snapshot/name)
    if a.pipeline_test:
        raw = np.random.randint(0, 256, (170,32,32,3), dtype=np.uint8)
        labels = np.repeat(np.arange(10), 17)
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
    state = initial_state(a.device)
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
    for task in range(next_task, 5):
        first = next_epoch if task == next_task else 0
        for epoch in range(first, epochs):
            start = time.monotonic()
            order = np.random.permutation(indices[task])
            with trace_path.open('a', buffering=1) as trace:
                for batch, pos in enumerate(range(0, len(order), 32)):
                    ix = order[pos:pos+32]
                    state, row = step(model, controller, optimizer, mixer_optimizer, memory,
                                      state, raw[ix], labels[ix], a.device, 3,mixer=a.mixer,kind=a.consistency,temperature=a.temperature)
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
            if a.stop_after_epochs and task*epochs+epoch+1>=a.stop_after_epochs:
                return
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


