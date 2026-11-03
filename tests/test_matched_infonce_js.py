import copy,json,sys,tempfile,subprocess,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproduce'))
import matched_infonce_js as m
import matched_mixing as ref

class MixedObjectiveTests(unittest.TestCase):
    def build(self,module=m):
        module.setup(17,'cuda');net=module.SmallCNN().cuda();ctl=module.Controller().cuda()
        return net,ctl,torch.optim.SGD(net.parameters(),lr=.03),torch.optim.SGD(ctl.parameters(),lr=.05),module.Reservoir(),module.initial_state('cuda')
    def data(self):
        return np.random.default_rng(4).integers(0,256,(4,32,32,3),dtype=np.uint8),np.array([0,1,0,1])
    def test_only_outer_objective_changed(self):
        raw,y=self.data();original_outer=ref.outer_objective
        for outer in ['infonce','js']:
            expected=[]
            def swapped(*args,**kwargs):
                args=list(args);args[9]=outer
                return original_outer(*args,**kwargs)
            with patch.object(ref,'outer_objective',swapped):
                net,ctl,opt,mo,mem,state=self.build(ref)
                for b in (4,2):
                    state,row=ref.step(net,ctl,opt,mo,mem,state,raw[:b],y[:b],'cuda',mixer='learned',kind='infonce')
                    expected.append(copy.deepcopy((net.state_dict(),ctl.state_dict(),state,row,mem.logits[:mem.size])))
            net,ctl,opt,mo,mem,state=self.build()
            self.assertEqual(sum(p.numel() for p in net.parameters()),288746)
            for b,exp in zip((4,2),expected):
                bn=next(v for v in net.modules() if isinstance(v,torch.nn.BatchNorm2d));before=bn.num_batches_tracked.item()
                state,row=m.step(net,ctl,opt,mo,mem,state,raw[:b],y[:b],'cuda',kind='infonce',outer_kind=outer)
                self.assertEqual(bn.num_batches_tracked.item()-before,1 if b==4 else 2)
                self.assertEqual(row.pop('outer_consistency'),outer);self.assertEqual(row,exp[3])
                for actual,expected_dict in zip((net.state_dict(),ctl.state_dict()),exp[:2]):
                    for k,v in actual.items():self.assertTrue(torch.equal(v,expected_dict[k]),k)
                for a,c in zip(state,exp[2]):self.assertTrue(torch.equal(a,c))
                self.assertTrue(torch.equal(mem.logits[:mem.size],exp[4]))
                self.assertGreater(row['mixer_updates'][-1]['grad_norm'],0)
                self.assertGreater(ctl.head.weight.grad[3].abs().sum().item(),0)

    def test_exact_resume(self):
        raw,y=self.data();net,ctl,opt,mo,mem,state=self.build()
        state,row=m.step(net,ctl,opt,mo,mem,state,raw,y,'cuda')
        saved=copy.deepcopy((net.state_dict(),ctl.state_dict(),opt.state_dict(),mo.state_dict(),mem,state,m.rng_state()))
        ns,row=m.step(net,ctl,opt,mo,mem,state,raw[:2],y[:2],'cuda');weights=copy.deepcopy(net.state_dict())
        net.load_state_dict(saved[0]);ctl.load_state_dict(saved[1]);opt.load_state_dict(saved[2]);mo.load_state_dict(saved[3]);mem=saved[4];state=saved[5];m.restore_rng(saved[6])
        ns2,row2=m.step(net,ctl,opt,mo,mem,state,raw[:2],y[:2],'cuda');self.assertEqual(row,row2)
        for a,b in zip(ns,ns2):self.assertTrue(torch.equal(a,b))
        for k,v in weights.items():self.assertTrue(torch.equal(v,net.state_dict()[k]))

    def test_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            cmd=[sys.executable,str(ROOT/'reproduce/matched_infonce_js.py'),'--pipeline-test','--output',d]
            for extra in (['--stop-after-epochs','2'],[]):
                p=subprocess.run(cmd+extra,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            trace=(Path(d)/'trace.jsonl').read_bytes()
            p=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(trace,(Path(d)/'trace.jsonl').read_bytes());self.assertEqual(len(trace.splitlines()),10)
            self.assertTrue((Path(d)/'m_progress.png').exists())
            self.assertEqual(json.loads((Path(d)/'complete.json').read_text())['presentations'],170)
            for line in trace.splitlines():
                row=json.loads(line);self.assertEqual(row['outer_consistency'],'js');self.assertEqual(row['consistency'],'infonce')
if __name__=='__main__':unittest.main()


