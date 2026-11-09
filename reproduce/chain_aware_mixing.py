"""Chain-aware weights: fixed m=.5, independently selected seen/unseen consistency."""
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
from report_chain_aware import report
import shutil

from libs.models import SmallCNN, backbone_metadata

from libs.models import ChainAwareController as Controller

@torch.no_grad()
def candidate_descriptors(model, x, branches):
    """One eval-mode joint forward; no committed BN updates or encoder gradients."""
    was_training = model.training
    model.eval()
    try:
        z,h = forward(model,torch.cat((x,*[v for view in branches for v in view])))
        zs,hs = z.chunk(7),h.chunk(7)
        desc = [torch.cat((hs[k]-hs[0],zs[k].softmax(-1)-zs[0].softmax(-1)),dim=-1)
                for k in range(1,7)]
        return torch.stack(desc).reshape(2,3,len(x),model.feature_dimension + 10).permute(0,2,1,3).contiguous()
    finally:
        model.train(was_training)


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

from libs.loss import js

from libs.loss import contrastive_scores

from libs.loss import infonce_loss


from libs.loss import consistency_loss, seen_unseen_feature_loss

# Extra random mixture draws never consume the historical shuffle/operator RNG.
MIXTURE_RNG = np.random.default_rng(170501)

def random_weights(n,device):
    m=torch.tensor(MIXTURE_RNG.beta(1,1,(2,n)),dtype=torch.float32,device=device)
    w=torch.tensor(MIXTURE_RNG.dirichlet(np.ones(3),(2,n)),dtype=torch.float32,device=device)
    return m,w

def mixed_views(x,branches,m,w):
    return [blend(x,v,m[i] if m.ndim==2 else m,w[i] if w.ndim==3 else w) for i,v in enumerate(branches)]

def chains(raw, pool, device, views_count=2, branch_count=3):
    views = []
    for _ in range(views_count):
        branches = []
        for _ in range(branch_count):
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

def initial_state(device, current_capacity=32, replay_capacity=32):
    # Current and replay positions occupy disjoint persistent slots.
    slots = current_capacity + replay_capacity
    return (torch.randn(slots,5,device=device),torch.randn(slots,5,device=device))

def active_slots(b,r,device,current_capacity=32):
    return torch.cat((torch.arange(b,device=device),current_capacity+torch.arange(r,device=device)))

def outer_objective(model,controller,context,incoming,x,branches,params,buffers,labels,kind='js',temperature=.1,candidates=None,target_features=None):
    assert not model.training
    om,ow,_=controller(context,incoming,candidates)
    joined=torch.cat((x,*mixed_views(x,branches,om,ow)))
    z,h=forward(model,joined,params,buffers)
    zs=list(z.chunk(3));hs=list(h.chunk(3))
    zs[0]=zs[0].detach();hs[0]=hs[0].detach()
    ce=F.cross_entropy(zs[0],labels)
    if kind in ('feature_match', 'seen_js_feature_match'):
        if target_features is None:
            raise ValueError('feature_match requires independent unseen targets')
        penalty = seen_unseen_feature_loss(torch.stack(hs[1:]), target_features)
        total = penalty + js(*zs) if kind == 'seen_js_feature_match' else penalty
        return total,penalty,om,ow,ce
    penalty=consistency_loss(zs,hs,labels,kind,temperature)
    return ce+penalty,penalty,om,ow,ce

def step(model,controller,optimizer,mixer_optimizer,memory,state,raw,labels,device,j=3,debug=False,mixer='learned',kind='infonce',temperature=.1,outer_kind='js'):
    return BiLevelTrainer.update_chain_aware_mixing(model, controller, optimizer, mixer_optimizer, memory, state, raw, labels, device, j, debug, mixer, kind, temperature, outer_kind=outer_kind, active_slots=active_slots, candidate_descriptors=candidate_descriptors, chains=chains, clone_buffers=clone_buffers, consistency_loss=consistency_loss, context_for=context_for, forward=forward, mixed_views=mixed_views, ops=ops, outer_objective=outer_objective, tensor=tensor)

def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)

def source_hashes():
    base=Path(__file__).parent
    names=['chain_aware_mixing.py','report_chain_aware.py','augmix_operations.py','AUGMIX_LICENSE',
           'download_cifar100_benchmark.py','download_cifar10_benchmark.py']
    return {name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in names} | library_source_hashes()

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

def main(*, bilevel_step=None, split_indices=None):
    p = argparse.ArgumentParser()
    p.add_argument('--mixer',choices=['learned'],default='learned')
    p.add_argument('--consistency',choices=['infonce','js'],default='infonce')
    if bilevel_step:
        p.set_defaults(outer_consistency='virtual_query_ce')
        p.add_argument('--batch-size', type=int, default=256)
    else:
        p.add_argument('--outer-consistency',choices=['infonce','js','feature_match','seen_js_feature_match'],default='js')
    p.add_argument('--temperature',type=float,default=.1)
    p.add_argument('--device', choices=['cuda'], default='cuda')
    p.add_argument('--root', type=Path, default=Path('experiments/metamix_cifar10/data'))
    p.add_argument('--output', type=Path, default=Path('experiments/virtual_mixer/seed17' if bilevel_step else 'experiments/chain_aware_mixing/seed17'))
    p.add_argument('--epochs-per-task', type=int, default=10)
    p.add_argument('--stop-after-epochs',type=int,default=0)
    p.add_argument('--pipeline-test', action='store_true')
    p.add_argument('--backbone', choices=['smallcnn', 'resnet18'], default='smallcnn')
    a = p.parse_args()
    batch_size = a.batch_size if bilevel_step else 32
    if batch_size < 1:
        p.error('batch-size must be positive')
    if not np.isfinite(a.temperature) or a.temperature<=0:p.error('positive finite temperature required')
    if a.epochs_per_task < 1:
        p.error('--epochs-per-task must be positive')
    if a.stop_after_epochs and not a.pipeline_test:
        p.error('stop-after-epochs is validation only')
    setup(17, a.device)
    a.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = a.output/'MetaMix'/'checkpoint.pt'
    checkpoint_path.parent.mkdir(exist_ok=True)
    classifier_objective = 'class-aware InfoNCE' if a.consistency == 'infonce' else 'JS'
    mixer_objective = 'class-aware InfoNCE' if a.outer_consistency == 'infonce' else 'JS'
    config = dict(method=f'Chain-aware fixed-m mixer: seen {classifier_objective} + unseen {mixer_objective} + DER++',mixer=a.mixer,consistency=a.consistency,temperature=a.temperature,consistency_coefficient=1.,
        fixed_m=.5,outer_consistency=a.outer_consistency,outer_coefficient=1.,candidate_features='128 raw feature deltas and 10 softmax probability deltas; detached eval joint7N forward',scorer='shared Linear(143,32),ReLU,Linear(32,1); zero final layer; softmax per image/view',random_mixture='none',
        sampling='preserve legacy global shuffle/ops and reservoir RNG; isolate mixture draws; generate both seen/unseen in all variants',
        matching_limit='new method: fixed m, candidate-conditioned per-view weights, selectable unseen consistency, extra eval descriptor passes', implementation='joint three-view forwards; replay augmentation; per-example slot state; pre-solver state advance',
        solver=f'Algorithm 2 explicit fixed-updated-classifier partial gradient; unseen {mixer_objective} drives mixer; original CE contributes zero mixer gradient; no unrolled hypergradient; not an explicit retention outer objective',
        context_alignment='broadcast batch dot/replay CE; individual er,et cyclic positional pairing for unequal batch sizes; zero er when empty',
        recurrent_state='64 persistent batch slots: current0:32 replay32:64; inactive slots unchanged; detach between batches; save pre-solver transition',
        bn='joint clean+two seen views train forward then separate distillation; context cloned buffers; joint clean+two unseen outer eval forward; candidate descriptors joint original+six chains eval before classifier and after classifier for unseen',
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
    if a.outer_consistency in ('feature_match', 'seen_js_feature_match'):
        config.update(
            method=f'Chain-aware fixed-m mixer: seen {classifier_objective} + seen-to-unseen feature MSE + DER++',
            mixer_objective='seen-to-unseen raw pooled feature MSE',
            target_policy='six unmixed unseen chains; each seen view matches its three unseen targets; detached; reused for J steps',
            feature_match_coefficient=1.,
            classifier_seen_consistency=a.consistency,
            classifier_seen_consistency_coefficient=1.,
            unseen_targets_use_mixer=False,
            unseen_targets_update_classifier=False,
            matching_limit='feature matching may average incompatible targets; fixed m=.5 restricts reachability; no robustness guarantee',
            solver='fixed updated classifier; gradients through seen mixtures only; no hypergradient; no outer clean CE term',
            bn='classifier joint clean+seen train; outer eval; seen descriptors recomputed after classifier update; six unseen target features cached without gradients')
    if a.outer_consistency == 'seen_js_feature_match':
        config['method'] += ' + mixer seen-view JS'
        config['mixer_objective'] = 'seen-view prediction JS + seen-to-unseen raw pooled feature MSE'
        config['mixer_seen_js_coefficient'] = 1.
    if bilevel_step:
        config.update(batch=batch_size, query_batch=32,
                      recurrent_state=f'{batch_size} current slots followed by 32 replay slots',
                      method=f'One-step bilevel mixer: seen {classifier_objective} + DER++',
                      J=1, mixer_objective='query clean CE + query unseen CE',
                      solver='differentiate through one plain classifier SGD update',
                      sampling='seed-17 stratified 10% query holdout per task; query sampled without replacement per batch',
                      matching_limit='one-step approximation on training-set query holdout',
                      bn='one committed support training pass plus distillation; virtual query in eval mode',
                      implementation='one virtual SGD step committed using detached support gradients',
                      query_unseen_chains=1, query_unseen_coefficient=1.)
        config['source_hashes']['virtual_mixer.py'] = hashlib.sha256(
            Path(__file__).with_name('virtual_mixer.py').read_bytes()).hexdigest()
    config.update(backbone_metadata(a.backbone))
    config['candidate_features'] = f"{config['feature_dimension']} raw feature deltas and 10 softmax probability deltas; detached eval joint7N forward"
    config['scorer'] = f"shared Linear({5 + config['feature_dimension'] + 10},32),ReLU,Linear(32,1); zero final layer; softmax per image/view"
    config['context_size'] = 2 * config['feature_dimension'] + 2
    config_path = a.output/'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError('Config/source mismatch: use a fresh output directory')
    else:
        atomic_json(config_path, config)
        snapshot=a.output/'source';snapshot.mkdir(exist_ok=True)
        for name in config['source_hashes']:
            snapshot_source(Path(__file__).parent, snapshot, name)
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
    if bilevel_step:
        indices, query_indices = split_indices(labels, config['tasks'])
        atomic_json(a.output/'query_indices.json', query_indices)
    index_path = a.output/'train_indices.json'
    if index_path.exists():
        assert json.loads(index_path.read_text()) == indices
    else:
        atomic_json(index_path, indices)
    model, controller = SmallCNN(backbone=a.backbone).to(a.device), Controller(feature_dimension=config['feature_dimension']).to(a.device)
    assert sum(p.numel() for p in model.parameters()) == config['parameters']
    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    mixer_optimizer = torch.optim.SGD(controller.parameters(), lr=.05)
    memory = Reservoir()
    state = initial_state(a.device, batch_size, config['replay_batch'])
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
    trainer = BiLevelTrainer(model, controller, optimizer, mixer_optimizer, memory, state, a.device, update=bilevel_step or step)
    for task in range(next_task, 5):
        first = next_epoch if task == next_task else 0
        for epoch in range(first, epochs):
            start = time.monotonic()
            order = np.random.permutation(indices[task])
            with trace_path.open('a', buffering=1) as trace:
                for batch, pos in enumerate(range(0, len(order), batch_size)):
                    ix = order[pos:pos+batch_size]
                    extra = {}
                    if bilevel_step:
                        pool = query_indices[task]
                        qix = np.random.choice(pool, min(config['query_batch'], len(pool)), replace=False)
                        extra['query_batch'] = raw[qix], labels[qix]
                        extra['current_capacity'] = batch_size
                    row = trainer.step(raw[ix], labels[ix], mixer=a.mixer, kind=a.consistency, temperature=a.temperature, outer_kind=a.outer_consistency, j=config['J'], **extra)
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
            if a.stop_after_epochs and task*epochs+epoch+1>=a.stop_after_epochs:
                return
    if not a.pipeline_test:
        assert updates == epochs * sum((len(ids)+batch_size-1)//batch_size for ids in indices)
        assert memory.seen == epochs * sum(map(len, indices))
    report(a.output, updates)
    if not (a.output/'evaluation.json').exists():
        evaluate(model, a.root, a.output, a.device, (raw, labels) if a.pipeline_test else None)
    atomic_json(a.output/'complete.json', dict(updates=updates, presentations=memory.seen,
        completed=time.time(), pipeline_test=a.pipeline_test))
    print('Completed training and evaluation', flush=True)

if __name__ == '__main__':
    main()


