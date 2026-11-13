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
import matched_mixing as m
import metamix_jointbn as ref
from libs.loss import contrastive_scores, infonce_loss


class MatchedTests(unittest.TestCase):
    def build(self,module):
        module.setup(17,'cuda');net=module.SmallCNN().cuda();ctl=module.Controller().cuda()
        return net,ctl,torch.optim.SGD(net.parameters(),lr=.03),torch.optim.SGD(ctl.parameters(),lr=.05),module.Reservoir(),module.initial_state('cuda')
    def data(self):
        return np.random.default_rng(4).integers(0,256,(4,32,32,3),dtype=np.uint8),np.array([0,1,0,1])
    def test_matching_and_reference(self):
        raw,y=self.data();fingerprints=[];legacy=None
        for mixer,kind in [('learned','js'),('random','js'),('random','infonce'),('learned','infonce')]:
            net,ctl,opt,mo,mem,state=self.build(m);rows=[];before=copy.deepcopy(ctl.state_dict())
            for b in (4,2):
                bn=next(v for v in net.modules() if isinstance(v,torch.nn.BatchNorm2d));tracked=bn.num_batches_tracked.item();had_replay=bool(mem.size)
                state,row,d=m.step(net,ctl,opt,mo,mem,state,raw[:b],y[:b],'cuda',mixer=mixer,kind=kind,debug=True)
                self.assertEqual(bn.num_batches_tracked.item()-tracked,1+int(had_replay));rows.append(row)
                self.assertFalse(d['stored'].requires_grad)
                if mixer=='random':
                    self.assertFalse(d['m'].requires_grad);self.assertFalse(d['w'].requires_grad)
                    self.assertEqual(d['m'].shape,(2,row['augmented_batch']))
                    torch.testing.assert_close(d['w'].sum(-1),torch.ones_like(d['m']))
                    self.assertFalse(torch.equal(d['m'][0],d['m'][1]))
                    self.assertEqual(row['mixer_updates'],[])
                else:self.assertEqual(len(row['mixer_updates']),3)
            fingerprints.append([r['sampling_sha256'] for r in rows])
            if mixer=='random':
                for k,v in ctl.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
            if (mixer,kind)==('learned','js'):legacy=copy.deepcopy((net.state_dict(),ctl.state_dict(),rows,mem.logits))
        for f in fingerprints:self.assertEqual(f,fingerprints[0])
        net,ctl,opt,mo,mem,state=self.build(ref);rows=[]
        for b in (4,2):state,row=ref.step(net,ctl,opt,mo,mem,state,raw[:b],y[:b],'cuda');rows.append(row)
        for actual,expected in zip((net.state_dict(),ctl.state_dict()),legacy[:2]):
            for k,v in actual.items():self.assertTrue(torch.equal(v,expected[k]),k)
        self.assertTrue(torch.equal(mem.logits[:mem.size],legacy[3][:mem.size]))
        for old,new in zip(rows,legacy[2]):self.assertEqual(old['classifier_loss'],new['classifier_loss'])
    def test_mask_and_resume(self):
        raw,y=self.data()
        h=[torch.randn(4,128,device='cuda',requires_grad=True) for _ in range(3)];labels=torch.tensor(y,device='cuda')
        scores,pos=contrastive_scores(*h,labels=labels)
        self.assertTrue(torch.isneginf(scores[0,2]));self.assertTrue(torch.isfinite(scores[0,1]));self.assertTrue(pos[0,4])
        loss=infonce_loss(*h,labels=labels);loss.backward()
        for v in h:self.assertGreater(v.grad.norm().item(),0)
        for mixer in ('random','learned'):
            net,ctl,opt,mo,mem,state=self.build(m)
            state,_=m.step(net,ctl,opt,mo,mem,state,raw,y,'cuda',mixer=mixer,kind='infonce')
            saved=copy.deepcopy((net.state_dict(),ctl.state_dict(),opt.state_dict(),mo.state_dict(),mem,state,m.rng_state()))
            ns,row=m.step(net,ctl,opt,mo,mem,state,raw[:2],y[:2],'cuda',mixer=mixer,kind='infonce');weights=copy.deepcopy(net.state_dict())
            net.load_state_dict(saved[0]);ctl.load_state_dict(saved[1]);opt.load_state_dict(saved[2]);mo.load_state_dict(saved[3]);mem=saved[4];state=saved[5];m.restore_rng(saved[6])
            ns2,row2=m.step(net,ctl,opt,mo,mem,state,raw[:2],y[:2],'cuda',mixer=mixer,kind='infonce')
            self.assertEqual(row,row2)
            for a,b in zip(ns,ns2):self.assertTrue(torch.equal(a,b))
            for k,v in weights.items():self.assertTrue(torch.equal(v,net.state_dict()[k]))
    def test_all_queued_pipelines(self):
        for mixer,kind in [('random','js'),('random','infonce'),('learned','infonce')]:
            with tempfile.TemporaryDirectory() as d:
                cmd=[sys.executable,str(ROOT/'reproduce/matched_mixing.py'),'--pipeline-test','--mixer',mixer,'--consistency',kind,'--output',d]
                for extra in (['--stop-after-epochs','2'],[]):
                    p=subprocess.run(cmd+extra,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
                trace=(Path(d)/'trace.jsonl').read_bytes()
                p=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
                self.assertEqual(trace,(Path(d)/'trace.jsonl').read_bytes());self.assertEqual(len(trace.splitlines()),10)
                self.assertTrue((Path(d)/'m_progress.png').exists())
                self.assertEqual(json.loads((Path(d)/'complete.json').read_text())['presentations'],170)
if __name__=='__main__':unittest.main()

