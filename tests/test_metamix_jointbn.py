import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproduce'))
import metamix_jointbn as m


class RevisedTests(unittest.TestCase):
    def build(self):
        m.setup(17,'cuda');net=m.SmallCNN().cuda();ctl=m.Controller().cuda()
        return net,ctl,torch.optim.SGD(net.parameters(),lr=.03),torch.optim.SGD(ctl.parameters(),lr=.05),m.Reservoir(),m.initial_state('cuda')
    def test_per_example_blending_context_and_timing(self):
        net,ctl,opt,mo,mem,state=self.build()
        raw=np.random.default_rng(7).integers(0,256,(4,32,32,3),dtype=np.uint8);y=np.array([0,1,0,1])
        x=m.tensor(raw,'cuda');branches=torch.rand(3,*x.shape,device='cuda')
        w=torch.tensor([[1.,0,0],[0,1.,0],[0,0,1.],[.2,.3,.5]],device='cuda')
        self.assertTrue(torch.equal(m.blend(x,branches,torch.ones(4,device='cuda'),w),x))
        mixed=m.blend(x,branches,torch.zeros(4,device='cuda'),w)
        for i in range(4):torch.testing.assert_close(mixed[i],sum(w[i,k]*branches[k,i] for k in range(3)))
        self.assertEqual(sum(p.numel() for p in net.parameters()),288746)
        for b in (4,2):
            before=tuple(v.clone() for v in state)
            state,row,d=m.step(net,ctl,opt,mo,mem,state,raw[:b],y[:b],'cuda',debug=True)
            n=row['augmented_batch'];self.assertEqual(n,b+row['replay_batch'])
            self.assertEqual(d['context'].shape,(n,258))
            self.assertEqual(d['unseen'][0].shape,(3,n,3,32,32))
            self.assertEqual(len(row['m_after_per_example']),n)
            self.assertGreater((d['context'][0,130:]-d['context'][1,130:]).norm().item(),0)
            slots=d['slots'];inactive=torch.ones(64,dtype=torch.bool,device='cuda');inactive[slots]=False
            for actual,expected,old in zip(state,d['advanced'],before):
                self.assertTrue(torch.equal(actual[slots],expected));self.assertTrue(torch.equal(actual[inactive],old[inactive]))
                self.assertFalse(actual.requires_grad)
            self.assertFalse(d['stored'].requires_grad)
            self.assertTrue(all(p.grad is None for p in net.parameters()))
            self.assertTrue(all(np.isfinite(r['grad_norm']) for r in row['mixer_updates']))
            if row['replay_batch']:self.assertIsNotNone(row['seen_js_replay'])
        self.assertEqual(mem.seen,6);self.assertEqual(mem.size,6)
    def test_gradient_contract_and_exact_resume(self):
        net,ctl,opt,mo,mem,state=self.build()
        raw=np.random.default_rng(8).integers(0,256,(4,32,32,3),dtype=np.uint8);y=np.array([0,1,0,1])
        state,row,d=m.step(net,ctl,opt,mo,mem,state,raw,y,'cuda',debug=True)
        total,js,_,_,_=m.outer_objective(net,ctl,d['context'],d['incoming'],d['xa'],d['unseen'],d['params'],d['buffers'],d['labels_a'])
        self.assertFalse(d['outer_ce'].requires_grad)
        g1=torch.autograd.grad(total,tuple(ctl.parameters()),retain_graph=True);g2=torch.autograd.grad(js,tuple(ctl.parameters()))
        for a,b in zip(g1,g2):self.assertTrue(torch.equal(a,b))
        snapshot=copy.deepcopy((net.state_dict(),ctl.state_dict(),opt.state_dict(),mo.state_dict(),mem,state,m.rng_state()))
        next_state,row=m.step(net,ctl,opt,mo,mem,state,raw[:2],y[:2],'cuda')
        weights=copy.deepcopy((net.state_dict(),ctl.state_dict()))
        net.load_state_dict(snapshot[0]);ctl.load_state_dict(snapshot[1]);opt.load_state_dict(snapshot[2]);mo.load_state_dict(snapshot[3]);mem=snapshot[4];state=snapshot[5];m.restore_rng(snapshot[6])
        repeat,other=m.step(net,ctl,opt,mo,mem,state,raw[:2],y[:2],'cuda')
        self.assertEqual(row,other)
        for a,b in zip(next_state,repeat):self.assertTrue(torch.equal(a,b))
        for actual,expected in zip((net.state_dict(),ctl.state_dict()),weights):
            for k,v in actual.items():self.assertTrue(torch.equal(v,expected[k]))
    def test_joint_forwards_and_bn_counts(self):
        net,ctl,opt,mo,mem,state=self.build()
        raw=np.random.default_rng(5).integers(0,256,(4,32,32,3),dtype=np.uint8);y=np.array([0,1,0,1])
        bn=next(v for v in net.modules() if isinstance(v,torch.nn.BatchNorm2d))
        for b in (4,2):
            before=bn.num_batches_tracked.item();calls=[]
            hook=net.register_forward_pre_hook(lambda module,args:calls.append((module.training,len(args[0]))))
            state,row,d=m.step(net,ctl,opt,mo,mem,state,raw[:b],y[:b],'cuda',debug=True)
            hook.remove();n=row['augmented_batch'];r=row['replay_batch']
            self.assertEqual(bn.num_batches_tracked.item()-before,1+int(r>0))
            self.assertEqual(calls,[(True,b)]+([(True,r)] if r else [])+[(True,3*n)]+([(True,r)] if r else [])+[(False,3*n)]*4)
            buffers={k:v.clone() for k,v in net.named_buffers()}
            _,_,om,ow,_=m.outer_objective(net,ctl,d['context'],d['incoming'],d['xa'],d['unseen'],d['params'],d['buffers'],d['labels_a'])
            for k,v in net.named_buffers():self.assertTrue(torch.equal(v,buffers[k]))
            # Eval joint and separate forwards use identical running statistics.
            separate=[m.forward(net,q,d['params'],d['buffers'])[0] for q in [d['xa']]+[m.blend(d['xa'],v,om,ow) for v in d['unseen']]]
            joined=m.forward(net,torch.cat([d['xa']]+[m.blend(d['xa'],v,om,ow) for v in d['unseen']]),d['params'],d['buffers'])[0]
            torch.testing.assert_close(joined,torch.cat(separate),rtol=1e-4,atol=1e-5)
    def test_pipeline_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            cmd=[sys.executable,str(ROOT/'reproduce/metamix_jointbn.py'),'--pipeline-test','--output',d]
            for extra in (['--stop-after-epochs','2'],[]):
                p=subprocess.run(cmd+extra,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
            path=Path(d);trace=(path/'trace.jsonl').read_bytes()
            p=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(trace,(path/'trace.jsonl').read_bytes());self.assertEqual(len(trace.splitlines()),10)
            self.assertEqual(json.loads((path/'complete.json').read_text())['presentations'],170)
            self.assertTrue((path/'m_progress.png').exists());self.assertTrue((path/'source/metamix_jointbn.py').exists())
if __name__=='__main__':unittest.main()
