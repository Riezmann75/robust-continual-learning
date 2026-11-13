"""CPU/CUDA correctness tests; no dataset download or full-budget training."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproduce'))
import augce_cifar10 as m
import feature_mse_cifar10 as baseline


class AugmentedCETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device='cuda' if torch.cuda.is_available() else 'cpu'
        cls.raw=np.random.default_rng(7).integers(0,256,(4,32,32,3),dtype=np.uint8)
        cls.labels=np.array([0,1,0,1])

    def build(self,decoder=False):
        m.setup(17,self.device)
        model=m.Network(decoder=decoder).to(self.device)
        return model,torch.optim.SGD(model.parameters(),lr=.03),m.Reservoir(),m.Streams(17)

    def test_shapes_and_reconstruction_gradients(self):
        model,_,_,_=self.build(True)
        self.assertEqual(sum(p.numel() for p in model.parameters()),718413)
        self.assertEqual(sum(p.numel() for p in model.decoder.parameters()),429667)
        x=m.tensor(self.raw,self.device)
        z,h=model(x,features=True)
        self.assertEqual(tuple(z.shape),(4,10))
        self.assertEqual(tuple(h.shape),(4,128))
        recon=model.decoder(h)
        self.assertEqual(recon.shape,x.shape)
        self.assertTrue(bool(((recon>=0)&(recon<=1)).all()))
        F.mse_loss(recon,x).backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.net.parameters()),0)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.decoder.parameters()),0)
        self.assertTrue(all(p.grad is None for p in model.head.parameters()))

    def test_objective_targets_replay_and_bn(self):
        model,opt,mem,rng=self.build(True)
        captured=[]
        reconstructed=[]
        handle=model.register_forward_hook(lambda net,args,result:captured.append(
            (args[0].detach().clone(),tuple(t.detach().clone() for t in result))) if isinstance(result,tuple) else None)
        dh=model.decoder.register_forward_hook(lambda net,args,result:reconstructed.append(result.detach().clone()))
        row,debug=m.step(model,opt,mem,rng,self.raw,self.labels,self.device,debug=True)
        handle.remove();dh.remove()
        inputs,(logits,features)=captured[0]
        z0,z1,z2=logits.chunk(3)
        y=torch.tensor(self.labels,device=self.device)
        expected=.5*(F.cross_entropy(z1,y)+F.cross_entropy(z2,y))
        self.assertAlmostEqual(row['augmented_ce'],expected.item(),places=6)
        self.assertAlmostEqual(row['reconstruction_loss'],F.mse_loss(reconstructed[0],inputs).item(),places=6)
        self.assertAlmostEqual(row['total_loss'],row['current_ce']+row['augmented_ce']+row['reconstruction_loss'],places=5)
        self.assertTrue(torch.equal(mem.logits[:4],z0.cpu()))
        self.assertFalse(debug['stored'].requires_grad)
        bn=[b for b in model.modules() if isinstance(b,torch.nn.BatchNorm2d)]
        self.assertTrue(all(b.num_batches_tracked.item()==1 for b in bn))
        second=m.step(model,opt,mem,rng,self.raw[:2],self.labels[:2],self.device)
        self.assertEqual(second['paired_examples'],6)
        self.assertGreater(second['replay_ce'],0)
        self.assertTrue(all(b.num_batches_tracked.item()==3 for b in bn))
        self.assertEqual(mem.seen,6)

    def test_zero_augmented_ce_recovers_baseline(self):
        def run(module):
            module.setup(17,self.device)
            net=module.Network().to(self.device)
            opt=torch.optim.SGD(net.parameters(),lr=.03)
            mem=module.Reservoir();rng=module.Streams(17)
            for take in (4,2):
                kwargs={'lambda_aug':0.,'lambda_recon':0.} if module is m else {'lambda_feat':0.}
                module.step(net,opt,mem,rng,self.raw[:take],self.labels[:take],self.device,**kwargs)
            return net.state_dict()
        old,new=run(baseline),run(m)
        self.assertTrue(all(torch.equal(v,new[k]) for k,v in old.items()))

    def test_decoder_checkpoint_exact_next_step(self):
        net,opt,mem,rng=self.build(True)
        m.step(net,opt,mem,rng,self.raw,self.labels,self.device)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint=Path(temp)/'checkpoint.pt'
            m.save(checkpoint,dict(model=net.state_dict(),opt=opt.state_dict(),
                                  mem=mem.state_dict(),rng=m.rng_state(rng,self.device)))
            first=m.step(net,opt,mem,rng,self.raw,self.labels,self.device)
            weights=copy.deepcopy(net.state_dict())
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            net.load_state_dict(saved['model']);opt.load_state_dict(saved['opt'])
            mem.load_state_dict(saved['mem']);m.restore_rng(saved['rng'],rng,self.device)
            second=m.step(net,opt,mem,rng,self.raw,self.labels,self.device)
            self.assertEqual({k:v for k,v in first.items() if k!='step_seconds'},
                             {k:v for k,v in second.items() if k!='step_seconds'})
            self.assertTrue(all(torch.equal(v,weights[k]) for k,v in net.state_dict().items()))

    def test_five_task_pipeline_and_resume_both_variants(self):
        with tempfile.TemporaryDirectory() as temp:
            for decoder in (False,True):
                dest=Path(temp)/str(decoder)
                cmd=[sys.executable,str(ROOT/'reproduce/augce_cifar10.py'),'--device',self.device,
                     '--pipeline-test','--output',str(dest)]
                if decoder:cmd.append('--decoder')
                result=subprocess.run(cmd+['--stop-after-epochs','2'],cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                before=(dest/'trace.jsonl').read_bytes()
                self.assertEqual(len(before.splitlines()),10)
                result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(before,(dest/'trace.jsonl').read_bytes())
                config=json.loads((dest/'config.json').read_text())
                self.assertEqual(config['decoder'],decoder)
                config['lambda_aug']=2
                (dest/'config.json').write_text(json.dumps(config))
                rejected=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
                self.assertNotEqual(rejected.returncode,0)
                self.assertIn('Config/source mismatch',rejected.stderr)

if __name__=='__main__':
    unittest.main(verbosity=2)
