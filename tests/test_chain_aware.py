import copy,json,sys,tempfile,subprocess,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproduce'))
import chain_aware_mixing as m

class ChainAwareTests(unittest.TestCase):
    def build(self):
        m.setup(17,'cuda');net=m.SmallCNN().cuda();ctl=m.Controller().cuda()
        return net,ctl,torch.optim.SGD(net.parameters(),lr=.03),torch.optim.SGD(ctl.parameters(),lr=.05),m.Reservoir(),m.initial_state('cuda')
    def data(self):
        return np.random.default_rng(4).integers(0,256,(4,32,32,3),dtype=np.uint8),np.array([0,1,0,1])
    def test_descriptors_scoring_and_outer(self):
        net,ctl,opt,mo,mem,state=self.build();raw,y=self.data();x=m.tensor(raw,'cuda')
        self.assertEqual(sum(p.numel() for p in net.parameters()),288746)
        branches=[torch.stack((x,x.flip(-1),x*.5)),torch.stack((x*.7,x.flip(-2),x))]
        before=copy.deepcopy(net.state_dict());desc=m.candidate_descriptors(net,x,branches)
        self.assertEqual(desc.shape,(2,4,3,138));self.assertFalse(desc.requires_grad);self.assertTrue(net.training)
        self.assertTrue(torch.equal(desc[0,:,0],torch.zeros_like(desc[0,:,0])))
        for k,v in before.items():self.assertTrue(torch.equal(v,net.state_dict()[k]))
        net.eval();z,h=m.forward(net,x);az,ah=m.forward(net,branches[0][1])
        torch.testing.assert_close(desc[0,:,1,:128],ah-h,atol=1e-6,rtol=1e-4)
        torch.testing.assert_close(desc[0,:,1,128:],az.softmax(-1)-z.softmax(-1),atol=1e-6,rtol=1e-4)
        self.assertEqual(z.shape,(4,10));self.assertEqual(h.shape,(4,128))
        context=torch.randn(4,258,device='cuda');incoming=tuple(s[:4] for s in state)
        # Nonuniform scorer so equivariance cannot pass trivially through uniform initialization.
        with torch.no_grad():ctl.scorer[-1].weight.normal_(0,.2)
        mm,w,ns=ctl(context,incoming,desc);perm=torch.tensor([2,0,1],device='cuda')
        mm2,w2,ns2=ctl(context,incoming,desc[:,:,perm])
        self.assertTrue(torch.equal(mm,torch.full_like(mm,.5)))
        torch.testing.assert_close(w.sum(-1),torch.ones_like(mm));torch.testing.assert_close(w2,w[:,:,perm])
        for a,b in zip(ns,ns2):self.assertTrue(torch.equal(a,b))
        params={k:v.detach() for k,v in net.named_parameters()};buffers=m.clone_buffers(net)
        loss,penalty,om,ow,ce=m.outer_objective(net,ctl,context,incoming,x,branches,params,buffers,torch.tensor(y,device='cuda'),'js',.1,candidates=desc)
        zz,_=m.forward(net,torch.cat((x,*m.mixed_views(x,branches,om,ow))),params,buffers)
        torch.testing.assert_close(penalty,m.js(*zz.chunk(3)))
        self.assertFalse(ce.requires_grad);loss.backward()
        self.assertGreater(sum(p.grad.square().sum().item() for p in ctl.parameters() if p.grad is not None),0)
        self.assertTrue(all(p.grad is None for p in net.parameters()))
        # Altered candidate content changes its score, not merely its position.
        changed=desc.clone();changed[0,:,0]+=1
        self.assertFalse(torch.allclose(ctl(context,incoming,changed)[1],w))

    def test_infonce_outer_formula_and_gradients(self):
        net,ctl,opt,mo,mem,state=self.build();raw,y=self.data()
        x=m.tensor(raw,'cuda');labels=torch.tensor(y,device='cuda')
        # Include same-class distinct images so the class-aware mask matters.
        branches=[torch.stack((x,x.flip(-1),x*.5)),torch.stack((x*.7,x.flip(-2),x))]
        desc=m.candidate_descriptors(net,x,branches)
        context=torch.randn(4,258,device='cuda');incoming=tuple(s[:4] for s in state)
        net.eval();params={k:v.detach() for k,v in net.named_parameters()};buffers=m.clone_buffers(net)
        before=copy.deepcopy(net.state_dict())
        loss,penalty,om,ow,ce=m.outer_objective(net,ctl,context,incoming,x,branches,params,buffers,labels,'infonce',.1,candidates=desc)
        _,features=m.forward(net,torch.cat((x,*m.mixed_views(x,branches,om,ow))),params,buffers)
        torch.testing.assert_close(penalty,m.infonce_loss(*features.chunk(3),temperature=.1,labels=labels))
        scores,positive=m.contrastive_scores(*features.chunk(3),temperature=.1,labels=labels)
        classes=labels.repeat(3)
        excluded=classes[:,None].eq(classes[None,:]) & ~positive
        self.assertTrue(torch.isneginf(scores[excluded]).all())
        self.assertFalse(ce.requires_grad)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in net.parameters()))
        self.assertGreater(sum(p.grad.square().sum().item() for p in ctl.parameters() if p.grad is not None),0)
        for k,v in before.items():self.assertTrue(torch.equal(v,net.state_dict()[k]))

    def test_step_and_exact_resume(self):
        for outer in ('js','infonce'):
            with self.subTest(outer=outer):self.check_step_and_exact_resume(outer)

    def check_step_and_exact_resume(self,outer):
        net,ctl,opt,mo,mem,state=self.build();raw,y=self.data()
        old=copy.deepcopy(ctl.state_dict())
        bn=next(v for v in net.modules() if isinstance(v,torch.nn.BatchNorm2d))
        tracked=bn.num_batches_tracked.item()
        state,row,d=m.step(net,ctl,opt,mo,mem,state,raw,y,'cuda',kind='infonce',outer_kind=outer,debug=True)
        self.assertEqual(bn.num_batches_tracked.item()-tracked,1)
        self.assertEqual(row['m_before'],.5);self.assertEqual(row['m_after'],.5)
        self.assertEqual(len(row['mixer_updates']),3);self.assertEqual(row['outer_consistency'],outer)
        self.assertFalse(d['stored'].requires_grad)
        torch.testing.assert_close(mem.logits[:4],d['stored'].cpu(),rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(v,old[k]) for k,v in ctl.state_dict().items()))
        saved=copy.deepcopy((net.state_dict(),ctl.state_dict(),opt.state_dict(),mo.state_dict(),mem,state,m.rng_state()))
        tracked=bn.num_batches_tracked.item()
        ns,row=m.step(net,ctl,opt,mo,mem,state,raw[:2],y[:2],'cuda',kind='infonce',outer_kind=outer)
        self.assertEqual(bn.num_batches_tracked.item()-tracked,2)
        self.assertEqual(row['replay_batch'],4);self.assertEqual(mem.seen,6)
        self.assertTrue(all(u['grad_norm']>0 for u in row['mixer_updates']))
        expected=copy.deepcopy((net.state_dict(),ctl.state_dict()))
        net.load_state_dict(saved[0]);ctl.load_state_dict(saved[1]);opt.load_state_dict(saved[2]);mo.load_state_dict(saved[3]);mem=saved[4];state=saved[5];m.restore_rng(saved[6])
        ns2,row2=m.step(net,ctl,opt,mo,mem,state,raw[:2],y[:2],'cuda',kind='infonce',outer_kind=outer)
        self.assertEqual(row,row2)
        for a,b in zip(ns,ns2):self.assertTrue(torch.equal(a,b))
        for expected_dict,actual in zip(expected,(net.state_dict(),ctl.state_dict())):
            for k,v in expected_dict.items():self.assertTrue(torch.equal(v,actual[k]),k)
        mem.add(np.tile(raw,(130,1,1,1)),np.tile(y,130),torch.zeros(520,10))
        self.assertEqual(mem.size,500)

    def test_js_classifier_formula(self):
        net,ctl,opt,mo,mem,state=self.build();raw,y=self.data()
        captured=[]
        def capture(model,args,result):
            if model.training and len(args[0])==3*len(raw):
                captured.append(result[0].detach().clone())
        handle=net.register_forward_hook(capture)
        try:
            state,row=m.step(net,ctl,opt,mo,mem,state,raw,y,'cuda',kind='js')
        finally:
            handle.remove()
        self.assertEqual(len(captured),1)
        self.assertAlmostEqual(row['seen_consistency'],m.js(*captured[0].chunk(3)).item(),places=7)
        self.assertAlmostEqual(row['classifier_loss'],row['current_ce']+row['seen_consistency'],places=6)
        self.assertEqual(row['consistency'],'js')
        self.assertEqual(row['outer_consistency'],'js')
        self.assertEqual(row['m_after'],.5)
        self.assertTrue(all(u['grad_norm']>0 for u in row['mixer_updates']))

    def test_pipeline_resume(self):
        for kind,outer in (('infonce','js'),('js','js'),('infonce','infonce')):
            with self.subTest(consistency=kind,outer=outer):
                self.check_pipeline_resume(kind,outer)

    def check_pipeline_resume(self,kind,outer):
        with tempfile.TemporaryDirectory() as d:
            cmd=[sys.executable,str(ROOT/'reproduce/chain_aware_mixing.py'),'--pipeline-test','--consistency',kind,'--outer-consistency',outer,'--output',d]
            for extra in (['--stop-after-epochs','2'],[]):
                p=subprocess.run(cmd+extra,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            trace=(Path(d)/'trace.jsonl').read_bytes()
            p=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(trace,(Path(d)/'trace.jsonl').read_bytes());self.assertEqual(len(trace.splitlines()),10)
            self.assertTrue((Path(d)/'weights_progress.png').exists())
            self.assertEqual(json.loads((Path(d)/'complete.json').read_text())['presentations'],170)
            config=json.loads((Path(d)/'config.json').read_text())
            self.assertEqual(config['consistency'],kind)
            self.assertEqual(config['outer_consistency'],outer)
            self.assertIn('seen JS' if kind=='js' else 'seen class-aware InfoNCE',config['method'])
            progress=json.loads((Path(d)/'progress.json').read_text())
            self.assertEqual(progress['classifier_consistency'],kind)
            self.assertEqual(progress['outer_consistency'],outer)
            for line in trace.splitlines():
                row=json.loads(line);self.assertEqual(row['m_after'],.5)
                self.assertEqual(row['consistency'],kind)
                self.assertEqual(row['outer_consistency'],outer)
if __name__=='__main__':unittest.main()
