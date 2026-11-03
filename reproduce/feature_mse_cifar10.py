"""Feature MSE + DER++ (random AugMix views), independent of MetaMix training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libs.trainer import SingleLevelTrainer
from libs import library_source_hashes, snapshot_source
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import platform
import random
import shutil
import time
import numpy as np
from PIL import Image
import PIL
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.datasets import CIFAR10
import augmix_operations as ops
from download_cifar100_benchmark import CORRUPTIONS, verify_arrays
from report_feature_mse import report

METHOD = 'Feature MSE + DER++ (random AugMix views)'
# Copied unchanged from SmallCNN in metamix_long.py (architecture provenance in config).
from libs.models import SmallCNN

from libs.models import Network as Network

class Streams:
    names = ('insertion', 'replay_ce', 'replay_logit', 'shuffle', 'mixture')
    def __init__(self, seed):
        seeds = np.random.SeedSequence(seed).spawn(6)
        for name, child in zip(self.names, seeds[:5]):
            setattr(self, name, np.random.default_rng(child))
        self.operations = np.random.RandomState(int(seeds[5].generate_state(1)[0]))

    def state_dict(self):
        return dict(generators={name:getattr(self,name).bit_generator.state for name in self.names},
                    operations=self.operations.get_state())

    def load_state_dict(self, state):
        for name in self.names:
            getattr(self,name).bit_generator.state = state['generators'][name]
        self.operations.set_state(state['operations'])

    @contextmanager
    def operation_scope(self):
        # Official PIL operators use global np.random. Isolate and restore it.
        # Single-threaded augmentation only; never run concurrent scopes.
        previous = np.random.get_state()
        np.random.set_state(self.operations.get_state())
        try:
            yield
        finally:
            self.operations.set_state(np.random.get_state())
            np.random.set_state(previous)

class Reservoir:
    def __init__(self, capacity=500):
        self.capacity, self.size, self.seen = capacity, 0, 0
        self.x = torch.zeros(capacity,32,32,3,dtype=torch.uint8)
        self.y = torch.zeros(capacity,dtype=torch.long)
        self.logits = torch.zeros(capacity,10)

    def add(self, raw, labels, logits, insertion_rng):
        assert not logits.requires_grad
        for image, label, logit in zip(raw,labels,logits.detach().cpu()):
            slot = self.seen if self.seen < self.capacity else int(insertion_rng.integers(self.seen+1))
            self.seen += 1
            if slot < self.capacity:
                self.x[slot].copy_(torch.as_tensor(np.array(image,copy=True)))
                self.y[slot] = int(label)
                self.logits[slot].copy_(logit)
            self.size = min(self.capacity,self.seen)

    def sample(self, rng, count=32):
        idx = rng.choice(self.size,min(count,self.size),replace=False)
        return self.x[idx].numpy(), self.y[idx].numpy(), self.logits[idx].clone()

    def state_dict(self):
        return dict(capacity=self.capacity,size=self.size,seen=self.seen,
                    x=self.x,y=self.y,logits=self.logits)

    def load_state_dict(self, state):
        self.capacity,self.size,self.seen = state['capacity'],state['size'],state['seen']
        self.x,self.y,self.logits = (state[k].cpu() for k in ('x','y','logits'))

def tensor(raw, device='cpu'):
    return torch.from_numpy(np.array(raw,copy=True)).permute(0,3,1,2).contiguous().to(
        device=device,dtype=torch.float32)/255.

def blend(x, branches, m, w):
    # x: N,C,H,W; branches: N,3,C,H,W; independent per-image weights.
    return m[:,None,None,None]*x + (1-m[:,None,None,None])*(
        branches*w[:,:,None,None,None]).sum(1)

@torch.no_grad()
def random_augmix_views(raw, streams):
    n = len(raw)
    x = tensor(raw)
    weights_m = torch.from_numpy(streams.mixture.beta(1,1,size=(2,n)).astype(np.float32))
    weights_w = torch.from_numpy(streams.mixture.dirichlet(np.ones(3),size=(2,n)).astype(np.float32))
    views = []
    with streams.operation_scope():
        for view in range(2):
            branches = np.empty((n,3,32,32,3),dtype=np.uint8)
            for i,image in enumerate(raw):
                for k in range(3):
                    im = Image.fromarray(image)
                    for _ in range(int(np.random.randint(1,4))):
                        im = ops.augmentations[int(np.random.randint(9))](im,3)
                    branches[i,k] = np.array(im,copy=True)
            branch_tensor = torch.from_numpy(branches).permute(0,1,4,2,3).contiguous().float()/255.
            views.append(blend(x,branch_tensor,weights_m[view],weights_w[view]).contiguous())
    return views, weights_m, weights_w

from libs.loss import feature_loss

def step(model,optimizer,memory,streams,raw,labels,device,debug=False,lambda_feat=1.):
    return SingleLevelTrainer.update_feature_mse_cifar10(model, optimizer, memory, streams, raw, labels, device, debug, lambda_feat, feature_loss=feature_loss, random_augmix_views=random_augmix_views, tensor=tensor)

def atomic_json(path,value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2))
    os.replace(temp,path)

def save(path,value):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        torch.save(value,f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp,path)

def source_hashes():
    base = Path(__file__).parent
    names = ['feature_mse_cifar10.py','report_feature_mse.py','augmix_operations.py',
             'AUGMIX_LICENSE','download_cifar100_benchmark.py','download_cifar10_benchmark.py']
    return {name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in names} | library_source_hashes()

def setup(seed,device):
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable')
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device=='cuda':
        torch.cuda.manual_seed_all(seed)
    if device=='mps':
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False

def rng_state(streams,device):
    state = dict(streams=streams.state_dict(),python=random.getstate(),
                 numpy=np.random.get_state(),torch=torch.get_rng_state())
    if device=='cuda':
        state['cuda']=torch.cuda.get_rng_state_all()
    if device=='mps':
        state['mps']=torch.mps.get_rng_state()
    return state

def restore_rng(state,streams,device):
    streams.load_state_dict(state['streams'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if device=='cuda':
        torch.cuda.set_rng_state_all(state['cuda'])
    if device=='mps':
        torch.mps.set_rng_state(state['mps'])

def summarize(rows,task,epoch,global_epoch,seconds):
    scalar = ['current_ce','replay_ce','replay_logit_mse','feature_loss','total_loss',
              'current_clean_accuracy','clean_accuracy','augmented_accuracy','cosine_view1','cosine_view2',
              'encoder_grad_norm','head_grad_norm','m_mean','m_std','step_seconds']
    vector = ['raw_feature_norms','raw_feature_variance','normalized_feature_variance','w_mean','w_std']
    summary = dict(task=task,epoch=epoch,global_epoch=global_epoch,batches=len(rows),
        updates=rows[-1]['update'],presentations=rows[-1]['presentations'],seconds=seconds,
        summary_weighting='equal batch means; final short batch retained',
        metrics={k:float(np.mean([r[k] for r in rows])) for k in scalar})
    summary['metrics'].update({k:np.mean([r[k] for r in rows],axis=0).tolist() for k in vector})
    # m samples are independent over each example/view; this is the pooled sample distribution.
    n = sum(2*r['paired_examples'] for r in rows)
    mean = sum(2*r['paired_examples']*r['m_mean'] for r in rows)/n
    second = sum(2*r['paired_examples']*(r['m_std']**2+r['m_mean']**2) for r in rows)/n
    summary['sampling_m'] = dict(n=n,mean=mean,std=float(np.sqrt(max(0,second-mean**2))),
                               min=min(r['m_min'] for r in rows),max=max(r['m_max'] for r in rows))
    return summary

def evaluate(model,root,output,device,synthetic=None):
    model.eval()
    @torch.no_grad()
    def accuracy(raw,labels):
        correct=task_correct=0
        for start in range(0,len(raw),256):
            logits=model(tensor(raw[start:start+256],device))
            y=torch.tensor(np.array(labels[start:start+256]),device=device)
            correct+=(logits.argmax(1)==y).sum().item()
            mask=torch.arange(10,device=device)[None,:]//2 == y[:,None]//2
            task_correct+=(logits.masked_fill(~mask,-torch.inf).argmax(1)==y).sum().item()
        return dict(n=len(raw),class_il=correct/len(raw),task_il=task_correct/len(raw))
    if synthetic is not None:
        result=dict(validation_only=True,clean=accuracy(*synthetic))
    else:
        clean=CIFAR10(root,train=False,download=False)
        target=root/'CIFAR-10-C'
        verify_arrays(target,clean.targets)
        result=dict(method=METHOD,clean=accuracy(clean.data,clean.targets),corruptions={})
        for name in CORRUPTIONS:
            array=np.load(target/(name+'.npy'),mmap_mode='r',allow_pickle=False)
            result['corruptions'][name]={str(severity):accuracy(
                array[(severity-1)*10000:severity*10000],clean.targets) for severity in range(1,6)}
            atomic_json(output/'evaluation.partial.json',result)
            print('Evaluated',name,flush=True)
        values=[v for cells in result['corruptions'].values() for v in cells.values()]
        result['corrupted_mean']={k:float(np.mean([v[k] for v in values])) for k in ('class_il','task_il')}
        result['corrupted_examples']=sum(v['n'] for v in values)
        assert result['corrupted_examples']==750000
    atomic_json(output/'evaluation.json',result)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cuda','mps','cpu'],default='cuda')
    parser.add_argument('--epochs-per-task',type=int,default=10)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--lambda-feat',type=float,default=1.)
    parser.add_argument('--root',type=Path,default=Path('experiments/metamix_cifar10/data'))
    parser.add_argument('--output',type=Path,default=Path('experiments/feature_mse_cifar10/seed17'))
    parser.add_argument('--pipeline-test',action='store_true')
    parser.add_argument('--stop-after-epochs',type=int,default=0,help='Validation-only interruption at an epoch boundary')
    a=parser.parse_args()
    if not np.isfinite(a.lambda_feat) or a.lambda_feat<0:
        parser.error('--lambda-feat must be finite and nonnegative')
    if a.epochs_per_task<1:
        parser.error('Epochs must be positive')
    if a.stop_after_epochs and not a.pipeline_test:
        parser.error('--stop-after-epochs is restricted to validation')
    if not 0 <= a.seed < 2**32:
        parser.error('--seed must be in [0, 2**32)')
    setup(a.seed,a.device)
    streams=Streams(a.seed)
    a.output.mkdir(parents=True,exist_ok=True)
    config=dict(method=METHOD,seed=a.seed,epochs_per_task=1 if a.pipeline_test else a.epochs_per_task,
        images_per_class=17 if a.pipeline_test else 5000,batch=32,replay_batch=32,memory_capacity=500,
        tasks=[[0,1],[2,3],[4,5],[6,7],[8,9]],resolution=32,parameters=288746,
        optimizer='SGD',lr=.03,momentum=0,weight_decay=0,schedule='constant',
        alpha=.2,beta=.5,lambda_feat=a.lambda_feat,feature_dimension=128,
        feature_loss='mean over examples and two views, sum over coordinates; symmetric gradients',
        normalization_mean=[.4914,.4822,.4465],normalization_std=[.247,.243,.261],
        views=2,chains=3,depth=[1,2,3],severity=3,m_distribution='Beta(1,1)',
        w_distribution='Dirichlet(1,1,1)',weights='independent per example and view, freshly sampled',
        bn='joint original/view1/view2 forward, then separate replay-logit forward',
        rng_streams=list(Streams.names)+['operations'],dtype='float32',device=a.device,
        hardware=torch.cuda.get_device_name() if a.device=='cuda' else platform.machine(),
        platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,
        torchvision=torchvision.__version__,numpy=np.__version__,pillow=PIL.__version__,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),threads=4,
        deterministic_cudnn=True,tf32=False,pipeline_test=a.pipeline_test,
        root=str(a.root.resolve()),source_hashes=source_hashes(),
        architecture_provenance='SmallCNN copied unchanged from reproduce/metamix_long.py; normalization wrapper added')
    if not a.pipeline_test:
        verified=a.root/'CIFAR-10-C'/'verified.json'
        if not verified.exists():
            raise RuntimeError('Verify datasets before training')
        config['data_verification_sha256']=hashlib.sha256(verified.read_bytes()).hexdigest()
    config_path=a.output/'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError('Config/source mismatch; use a new output directory')
    else:
        atomic_json(config_path,config)
        snapshot=a.output/'source'
        snapshot.mkdir(exist_ok=True)
        for name in config['source_hashes']:
            snapshot_source(Path(__file__).parent, snapshot, name)
    if a.pipeline_test:
        raw=np.random.default_rng(900).integers(0,256,(170,32,32,3),dtype=np.uint8)
        labels=np.repeat(np.arange(10),17)
    else:
        data=CIFAR10(a.root,train=True,download=False)
        raw,labels=data.data,np.array(data.targets)
        assert len(raw)==50000 and all((labels==c).sum()==5000 for c in range(10))
    indices=[np.flatnonzero(np.isin(labels,pair)).tolist() for pair in config['tasks']]
    index_path=a.output/'train_indices.json'
    if index_path.exists():
        assert json.loads(index_path.read_text())==indices
    else:
        atomic_json(index_path,indices)
    model=Network().to(a.device)
    assert sum(p.numel() for p in model.parameters())==288746
    optimizer=torch.optim.SGD(model.parameters(),lr=.03)
    memory=Reservoir()
    next_task=next_epoch=updates=0
    epochs=[]
    checkpoint=a.output/'checkpoint.pt'
    if checkpoint.exists():
        # Trusted local checkpoint; CPU map keeps memory and RNG payloads on CPU.
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['config']==config
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        memory.load_state_dict(saved['memory'])
        restore_rng(saved['rng'],streams,a.device)
        next_task,next_epoch,updates=saved['next_task'],saved['next_epoch'],saved['updates']
        epochs=saved['epochs']
    trace=a.output/'trace.jsonl'
    if trace.exists():
        tmp=trace.with_suffix('.tmp')
        with trace.open() as src,tmp.open('w') as dst:
            for line in src:
                try:
                    row=json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row['update']<=updates:
                    dst.write(json.dumps(row)+'\n')
        os.replace(tmp,trace)
    atomic_json(a.output/'process.json',dict(pid=os.getpid(),started=time.time(),
        host=platform.node(),command=' '.join(__import__('sys').argv)))
    print(json.dumps(dict(event='start_or_resume',pid=os.getpid(),updates=updates,config=config)),flush=True)
    run_start=time.monotonic()
    per_task=config['epochs_per_task']
    trainer = SingleLevelTrainer(model, optimizer, memory, streams, a.device, update=step)
    for task in range(next_task,5):
        for epoch in range(next_epoch if task==next_task else 0,per_task):
            start=time.monotonic()
            order=streams.shuffle.permutation(indices[task])
            rows=[]
            with trace.open('a',buffering=1) as f:
                for batch,pos in enumerate(range(0,len(order),32)):
                    idx=order[pos:pos+32]
                    row = trainer.step(raw[idx], labels[idx], lambda_feat=a.lambda_feat)
                    updates+=1
                    row.update(task=task+1,epoch=epoch+1,global_epoch=task*per_task+epoch+1,
                        batch=batch,update=updates,time=time.time(),elapsed_session_seconds=time.monotonic()-run_start)
                    rows.append(row)
                    f.write(json.dumps(row)+'\n')
                f.flush()
                os.fsync(f.fileno())
            summary=summarize(rows,task+1,epoch+1,task*per_task+epoch+1,time.monotonic()-start)
            epochs.append(summary)
            nt,ne=(task+1,0) if epoch+1==per_task else (task,epoch+1)
            save(checkpoint,dict(config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),
                memory=memory.state_dict(),rng=rng_state(streams,a.device),next_task=nt,next_epoch=ne,
                updates=updates,epochs=epochs))
            report(a.output,epochs)
            print(json.dumps(dict(event='epoch',**summary)),flush=True)
            if a.stop_after_epochs and len(epochs)>=a.stop_after_epochs:
                print('Validation boundary stop',flush=True)
                return
    if not a.pipeline_test:
        assert updates==5*per_task*313 and memory.seen==5*per_task*10000
    report(a.output,epochs)
    if not (a.output/'evaluation.json').exists():
        evaluate(model,a.root,a.output,a.device,(raw,labels) if a.pipeline_test else None)
    atomic_json(a.output/'complete.json',dict(method=METHOD,updates=updates,presentations=memory.seen,
        completed=time.time(),pipeline_test=a.pipeline_test))
    print('Completed training and evaluation',flush=True)

if __name__=='__main__':
    main()
