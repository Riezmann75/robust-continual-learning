"""Single-level classifier and bi-level classifier/controller batch updates.

Runners supply data/augmentation and objective hooks explicitly. This preserves
their separate random streams and allows objectives to be tested in isolation.
The update_* methods also expose the original stateless batch interfaces.
"""
import hashlib
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call


class SingleLevelTrainer:
    """Own classifier training resources and call a configured batch update.

    ``update`` is a runner's compatible step function or an update_* method;
    pass that method's required hooks and loss coefficients in ``options``.
    """
    def __init__(self, model, optimizer, memory, streams, device, *, update, **options):
        self.model, self.optimizer = model, optimizer
        self.memory, self.streams, self.device = memory, streams, device
        self.update, self.options = update, options

    def step(self, raw, labels, **options):
        return self.update(self.model, self.optimizer, self.memory, self.streams,
                           raw, labels, self.device, **(self.options | options))

    @staticmethod
    def update_augce_cifar10(model,optimizer,memory,streams,raw,labels,device,debug=False,lambda_aug=1.,lambda_recon=1., *, random_augmix_views, tensor):
        start = time.monotonic()
        ce_sample = memory.sample(streams.replay_ce) if memory.size else None
        logit_sample = memory.sample(streams.replay_logit) if memory.size else None
        raw_a = np.concatenate((raw,ce_sample[0])) if ce_sample is not None else raw
        labels_a = np.concatenate((labels,ce_sample[1])) if ce_sample is not None else labels
        y = torch.as_tensor(labels_a,dtype=torch.long,device=device)
        x0 = tensor(raw_a,device)
        views,m,w = random_augmix_views(raw_a,streams)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        joint_inputs = torch.cat((x0,views[0].to(device),views[1].to(device)))
        z,h = model(joint_inputs,features=True)
        z0,z1,z2 = z.chunk(3)
        h0,h1,h2 = h.chunk(3)
        augmented_ce = .5*(F.cross_entropy(z1,y)+F.cross_entropy(z2,y))
        reconstruction = F.mse_loss(model.decoder(h),joint_inputs) if model.decoder is not None else z0.new_zeros(())
        current_ce = F.cross_entropy(z0[:len(raw)],y[:len(raw)])
        replay_ce = F.cross_entropy(z0[len(raw):],y[len(raw):]) if ce_sample is not None else z0.new_zeros(())
        replay_mse = z0.new_zeros(())
        if logit_sample is not None:
            replay_logits = model(tensor(logit_sample[0],device))
            replay_mse = F.mse_loss(replay_logits,logit_sample[2].to(device))
        stored = z0[:len(raw)].detach().cpu().clone()
        loss = current_ce+.5*replay_ce+.2*replay_mse+lambda_aug*augmented_ce+lambda_recon*reconstruction
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite loss or gradients')
        encoder_grad = torch.stack([p.grad.square().sum() for p in model.net.parameters()]).sum().sqrt().item()
        head_grad = torch.stack([p.grad.square().sum() for p in model.head.parameters()]).sum().sqrt().item()
        decoder_grad = (torch.stack([p.grad.square().sum() for p in model.decoder.parameters()
                        if p.grad is not None]).sum().sqrt().item() if model.decoder is not None else 0.)
        optimizer.step()
        memory.add(raw,labels,stored,streams.insertion)
        with torch.no_grad():
            normalized = [F.normalize(a,p=2,dim=1,eps=1e-8) for a in (h0,h1,h2)]
            row = dict(current_ce=current_ce.item(),replay_ce=replay_ce.item(),replay_logit_mse=replay_mse.item(),
                augmented_ce=augmented_ce.item(),reconstruction_loss=reconstruction.item(),
                weighted_augmented_ce=(lambda_aug*augmented_ce).item(),
                weighted_reconstruction=(lambda_recon*reconstruction).item(),
                decoder_grad_norm=decoder_grad,total_loss=loss.item(),
                current_clean_accuracy=(z0[:len(raw)].argmax(1)==y[:len(raw)]).float().mean().item(),
                clean_accuracy=(z0.argmax(1)==y).float().mean().item(),
                augmented_accuracy=((z1.argmax(1)==y).float().mean()+(z2.argmax(1)==y).float().mean()).item()/2,
                cosine_view1=(normalized[0]*normalized[1]).sum(1).mean().item(),
                cosine_view2=(normalized[0]*normalized[2]).sum(1).mean().item(),
                raw_feature_norms=[a.norm(dim=1).mean().item() for a in (h0,h1,h2)],
                raw_feature_variance=[a.var(dim=0,unbiased=False).mean().item() for a in (h0,h1,h2)],
                normalized_feature_variance=[a.var(dim=0,unbiased=False).mean().item() for a in normalized],
                encoder_grad_norm=encoder_grad,head_grad_norm=head_grad,
                m_mean=m.mean().item(),m_std=m.std(unbiased=False).item(),m_min=m.min().item(),m_max=m.max().item(),
                m_view_means=m.mean(1).tolist(),w_mean=w.mean((0,1)).tolist(),
                w_std=w.std((0,1),unbiased=False).tolist(),w_view_means=w.mean(1).tolist(),
                current_batch=len(raw),paired_examples=len(raw_a),replay_batch=0 if ce_sample is None else len(ce_sample[0]),
                presentations=memory.seen,memory_size=memory.size,step_seconds=time.monotonic()-start)
        if debug:
            return row,dict(stored=stored,raw_a=raw_a,m=m,w=w)
        return row

    @staticmethod
    def update_barlow_cifar10(model,optimizer,memory,streams,raw,labels,device,debug=False,lambda_aug=1.,lambda_recon=1.,lambda_barlow=1.,barlow_off_weight=.0051, *, barlow_loss, random_augmix_views, tensor):
        start = time.monotonic()
        ce_sample = memory.sample(streams.replay_ce) if memory.size else None
        logit_sample = memory.sample(streams.replay_logit) if memory.size else None
        raw_a = np.concatenate((raw,ce_sample[0])) if ce_sample is not None else raw
        labels_a = np.concatenate((labels,ce_sample[1])) if ce_sample is not None else labels
        y = torch.as_tensor(labels_a,dtype=torch.long,device=device)
        x0 = tensor(raw_a,device)
        views,m,w = random_augmix_views(raw_a,streams)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        joint_inputs = torch.cat((x0,views[0].to(device),views[1].to(device)))
        z,h = model(joint_inputs,features=True)
        z0,z1,z2 = z.chunk(3)
        h0,h1,h2 = h.chunk(3)
        barlow,barlow_metrics=barlow_loss(h1,h2,barlow_off_weight)
        augmented_ce = .5*(F.cross_entropy(z1,y)+F.cross_entropy(z2,y))
        reconstruction = F.mse_loss(model.decoder(h),joint_inputs) if model.decoder is not None else z0.new_zeros(())
        current_ce = F.cross_entropy(z0[:len(raw)],y[:len(raw)])
        replay_ce = F.cross_entropy(z0[len(raw):],y[len(raw):]) if ce_sample is not None else z0.new_zeros(())
        replay_mse = z0.new_zeros(())
        if logit_sample is not None:
            replay_logits = model(tensor(logit_sample[0],device))
            replay_mse = F.mse_loss(replay_logits,logit_sample[2].to(device))
        stored = z0[:len(raw)].detach().cpu().clone()
        loss = current_ce+.5*replay_ce+.2*replay_mse+lambda_aug*augmented_ce+lambda_recon*reconstruction+lambda_barlow*barlow
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite loss or gradients')
        encoder_grad = torch.stack([p.grad.square().sum() for p in model.net.parameters()]).sum().sqrt().item()
        head_grad = torch.stack([p.grad.square().sum() for p in model.head.parameters()]).sum().sqrt().item()
        decoder_grad = (torch.stack([p.grad.square().sum() for p in model.decoder.parameters()
                        if p.grad is not None]).sum().sqrt().item() if model.decoder is not None else 0.)
        optimizer.step()
        memory.add(raw,labels,stored,streams.insertion)
        with torch.no_grad():
            normalized = [F.normalize(a,p=2,dim=1,eps=1e-8) for a in (h0,h1,h2)]
            row = dict(current_ce=current_ce.item(),replay_ce=replay_ce.item(),replay_logit_mse=replay_mse.item(),
                augmented_ce=augmented_ce.item(),reconstruction_loss=reconstruction.item(),
                weighted_augmented_ce=(lambda_aug*augmented_ce).item(),
                weighted_reconstruction=(lambda_recon*reconstruction).item(),
                decoder_grad_norm=decoder_grad,total_loss=loss.item(),
                current_clean_accuracy=(z0[:len(raw)].argmax(1)==y[:len(raw)]).float().mean().item(),
                clean_accuracy=(z0.argmax(1)==y).float().mean().item(),
                augmented_accuracy=((z1.argmax(1)==y).float().mean()+(z2.argmax(1)==y).float().mean()).item()/2,
                cosine_view1=(normalized[0]*normalized[1]).sum(1).mean().item(),
                cosine_view2=(normalized[0]*normalized[2]).sum(1).mean().item(),
                raw_feature_norms=[a.norm(dim=1).mean().item() for a in (h0,h1,h2)],
                raw_feature_variance=[a.var(dim=0,unbiased=False).mean().item() for a in (h0,h1,h2)],
                normalized_feature_variance=[a.var(dim=0,unbiased=False).mean().item() for a in normalized],
                encoder_grad_norm=encoder_grad,head_grad_norm=head_grad,
                m_mean=m.mean().item(),m_std=m.std(unbiased=False).item(),m_min=m.min().item(),m_max=m.max().item(),
                m_view_means=m.mean(1).tolist(),w_mean=w.mean((0,1)).tolist(),
                w_std=w.std((0,1),unbiased=False).tolist(),w_view_means=w.mean(1).tolist(),
                current_batch=len(raw),paired_examples=len(raw_a),replay_batch=0 if ce_sample is None else len(ce_sample[0]),
                presentations=memory.seen,memory_size=memory.size,step_seconds=time.monotonic()-start)
        row.update(barlow_loss=barlow.item(),weighted_barlow=(lambda_barlow*barlow).item(),
                   **{k:v.item() for k,v in barlow_metrics.items()})
        if debug:
            return row,dict(stored=stored,raw_a=raw_a,m=m,w=w)
        return row

    @staticmethod
    def update_bt_threepair_cifar10(model,optimizer,memory,streams,raw,labels,device,debug=False,lambda_bt=1.,off_weight=.0051, *, random_augmix_views, tensor, three_pair_loss):
        start = time.monotonic()
        ce_sample = memory.sample(streams.replay_ce) if memory.size else None
        logit_sample = memory.sample(streams.replay_logit) if memory.size else None
        raw_a = np.concatenate((raw,ce_sample[0])) if ce_sample is not None else raw
        labels_a = np.concatenate((labels,ce_sample[1])) if ce_sample is not None else labels
        y = torch.as_tensor(labels_a,dtype=torch.long,device=device)
        x0 = tensor(raw_a,device)
        views,m,w = random_augmix_views(raw_a,streams)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z,h = model(torch.cat((x0,views[0].to(device),views[1].to(device))),features=True)
        z0,z1,z2 = z.chunk(3)
        h0,h1,h2 = h.chunk(3)
        bt,bt_metrics = three_pair_loss(h0,h1,h2,off_weight)
        current_ce = F.cross_entropy(z0[:len(raw)],y[:len(raw)])
        replay_ce = F.cross_entropy(z0[len(raw):],y[len(raw):]) if ce_sample is not None else z0.new_zeros(())
        replay_mse = z0.new_zeros(())
        if logit_sample is not None:
            replay_logits = model(tensor(logit_sample[0],device))
            replay_mse = F.mse_loss(replay_logits,logit_sample[2].to(device))
        stored = z0[:len(raw)].detach().cpu().clone()
        loss = current_ce+.5*replay_ce+.2*replay_mse+lambda_bt*bt
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite loss or gradients')
        encoder_grad = torch.stack([p.grad.square().sum() for p in model.net.parameters()]).sum().sqrt().item()
        head_grad = torch.stack([p.grad.square().sum() for p in model.head.parameters()]).sum().sqrt().item()
        optimizer.step()
        memory.add(raw,labels,stored,streams.insertion)
        with torch.no_grad():
            normalized = [F.normalize(a,p=2,dim=1,eps=1e-8) for a in (h0,h1,h2)]
            row = dict(current_ce=current_ce.item(),replay_ce=replay_ce.item(),replay_logit_mse=replay_mse.item(),
                bt_loss=bt.item(),weighted_bt=(lambda_bt*bt).item(),total_loss=loss.item(),
                current_clean_accuracy=(z0[:len(raw)].argmax(1)==y[:len(raw)]).float().mean().item(),
                clean_accuracy=(z0.argmax(1)==y).float().mean().item(),
                augmented_accuracy=((z1.argmax(1)==y).float().mean()+(z2.argmax(1)==y).float().mean()).item()/2,
                cosine_view1=(normalized[0]*normalized[1]).sum(1).mean().item(),
                cosine_view2=(normalized[0]*normalized[2]).sum(1).mean().item(),
                raw_feature_norms=[a.norm(dim=1).mean().item() for a in (h0,h1,h2)],
                raw_feature_variance=[a.var(dim=0,unbiased=False).mean().item() for a in (h0,h1,h2)],
                normalized_feature_variance=[a.var(dim=0,unbiased=False).mean().item() for a in normalized],
                encoder_grad_norm=encoder_grad,head_grad_norm=head_grad,
                m_mean=m.mean().item(),m_std=m.std(unbiased=False).item(),m_min=m.min().item(),m_max=m.max().item(),
                m_view_means=m.mean(1).tolist(),w_mean=w.mean((0,1)).tolist(),
                w_std=w.std((0,1),unbiased=False).tolist(),w_view_means=w.mean(1).tolist(),
                current_batch=len(raw),paired_examples=len(raw_a),replay_batch=0 if ce_sample is None else len(ce_sample[0]),
                presentations=memory.seen,memory_size=memory.size,step_seconds=time.monotonic()-start)
        row.update({k:v.item() for k,v in bt_metrics.items()})
        if debug:
            return row,dict(stored=stored,raw_a=raw_a,m=m,w=w)
        return row

    @staticmethod
    def update_feature_mse_cifar10(model,optimizer,memory,streams,raw,labels,device,debug=False,lambda_feat=1., *, feature_loss, random_augmix_views, tensor):
        start = time.monotonic()
        ce_sample = memory.sample(streams.replay_ce) if memory.size else None
        logit_sample = memory.sample(streams.replay_logit) if memory.size else None
        raw_a = np.concatenate((raw,ce_sample[0])) if ce_sample is not None else raw
        labels_a = np.concatenate((labels,ce_sample[1])) if ce_sample is not None else labels
        y = torch.as_tensor(labels_a,dtype=torch.long,device=device)
        x0 = tensor(raw_a,device)
        views,m,w = random_augmix_views(raw_a,streams)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z,h = model(torch.cat((x0,views[0].to(device),views[1].to(device))),features=True)
        z0,z1,z2 = z.chunk(3)
        h0,h1,h2 = h.chunk(3)
        feat = feature_loss(h0,h1,h2)
        current_ce = F.cross_entropy(z0[:len(raw)],y[:len(raw)])
        replay_ce = F.cross_entropy(z0[len(raw):],y[len(raw):]) if ce_sample is not None else z0.new_zeros(())
        replay_mse = z0.new_zeros(())
        if logit_sample is not None:
            replay_logits = model(tensor(logit_sample[0],device))
            replay_mse = F.mse_loss(replay_logits,logit_sample[2].to(device))
        stored = z0[:len(raw)].detach().cpu().clone()
        loss = current_ce+.5*replay_ce+.2*replay_mse+lambda_feat*feat
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite loss or gradients')
        encoder_grad = torch.stack([p.grad.square().sum() for p in model.net.parameters()]).sum().sqrt().item()
        head_grad = torch.stack([p.grad.square().sum() for p in model.head.parameters()]).sum().sqrt().item()
        optimizer.step()
        memory.add(raw,labels,stored,streams.insertion)
        with torch.no_grad():
            normalized = [F.normalize(a,p=2,dim=1,eps=1e-8) for a in (h0,h1,h2)]
            row = dict(current_ce=current_ce.item(),replay_ce=replay_ce.item(),replay_logit_mse=replay_mse.item(),
                feature_loss=feat.item(),total_loss=loss.item(),
                current_clean_accuracy=(z0[:len(raw)].argmax(1)==y[:len(raw)]).float().mean().item(),
                clean_accuracy=(z0.argmax(1)==y).float().mean().item(),
                augmented_accuracy=((z1.argmax(1)==y).float().mean()+(z2.argmax(1)==y).float().mean()).item()/2,
                cosine_view1=(normalized[0]*normalized[1]).sum(1).mean().item(),
                cosine_view2=(normalized[0]*normalized[2]).sum(1).mean().item(),
                raw_feature_norms=[a.norm(dim=1).mean().item() for a in (h0,h1,h2)],
                raw_feature_variance=[a.var(dim=0,unbiased=False).mean().item() for a in (h0,h1,h2)],
                normalized_feature_variance=[a.var(dim=0,unbiased=False).mean().item() for a in normalized],
                encoder_grad_norm=encoder_grad,head_grad_norm=head_grad,
                m_mean=m.mean().item(),m_std=m.std(unbiased=False).item(),m_min=m.min().item(),m_max=m.max().item(),
                m_view_means=m.mean(1).tolist(),w_mean=w.mean((0,1)).tolist(),
                w_std=w.std((0,1),unbiased=False).tolist(),w_view_means=w.mean(1).tolist(),
                current_batch=len(raw),paired_examples=len(raw_a),replay_batch=0 if ce_sample is None else len(ce_sample[0]),
                presentations=memory.seen,memory_size=memory.size,step_seconds=time.monotonic()-start)
        if debug:
            return row,dict(stored=stored,raw_a=raw_a,m=m,w=w)
        return row

    @staticmethod
    def update_infonce_cifar10(model,optimizer,memory,streams,raw,labels,device,debug=False,lambda_nce=1.,temperature=.1, *, contrastive_scores, infonce_loss, random_augmix_views, tensor):
        start = time.monotonic()
        ce_sample = memory.sample(streams.replay_ce) if memory.size else None
        logit_sample = memory.sample(streams.replay_logit) if memory.size else None
        raw_a = np.concatenate((raw,ce_sample[0])) if ce_sample is not None else raw
        labels_a = np.concatenate((labels,ce_sample[1])) if ce_sample is not None else labels
        y = torch.as_tensor(labels_a,dtype=torch.long,device=device)
        x0 = tensor(raw_a,device)
        views,m,w = random_augmix_views(raw_a,streams)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z,h = model(torch.cat((x0,views[0].to(device),views[1].to(device))),features=True)
        z0,z1,z2 = z.chunk(3)
        h0,h1,h2 = h.chunk(3)
        nce = infonce_loss(h0,h1,h2,temperature)
        current_ce = F.cross_entropy(z0[:len(raw)],y[:len(raw)])
        replay_ce = F.cross_entropy(z0[len(raw):],y[len(raw):]) if ce_sample is not None else z0.new_zeros(())
        replay_mse = z0.new_zeros(())
        if logit_sample is not None:
            replay_logits = model(tensor(logit_sample[0],device))
            replay_mse = F.mse_loss(replay_logits,logit_sample[2].to(device))
        stored = z0[:len(raw)].detach().cpu().clone()
        loss = current_ce+.5*replay_ce+.2*replay_mse+lambda_nce*nce
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite loss or gradients')
        encoder_grad = torch.stack([p.grad.square().sum() for p in model.net.parameters()]).sum().sqrt().item()
        head_grad = torch.stack([p.grad.square().sum() for p in model.head.parameters()]).sum().sqrt().item()
        optimizer.step()
        memory.add(raw,labels,stored,streams.insertion)
        with torch.no_grad():
            scores, positives = contrastive_scores(h0,h1,h2,temperature)
            nonself = ~torch.eye(len(scores),dtype=torch.bool,device=device)
            negatives = nonself & ~positives
            retrieval = positives[torch.arange(len(scores),device=device),scores.argmax(1)].float().mean().item()
            normalized = [F.normalize(a,p=2,dim=1,eps=1e-8) for a in (h0,h1,h2)]
            row = dict(current_ce=current_ce.item(),replay_ce=replay_ce.item(),replay_logit_mse=replay_mse.item(),
                infonce_loss=nce.item(),total_loss=loss.item(),
                current_clean_accuracy=(z0[:len(raw)].argmax(1)==y[:len(raw)]).float().mean().item(),
                clean_accuracy=(z0.argmax(1)==y).float().mean().item(),
                augmented_accuracy=((z1.argmax(1)==y).float().mean()+(z2.argmax(1)==y).float().mean()).item()/2,
                cosine_view1=(normalized[0]*normalized[1]).sum(1).mean().item(),
                cosine_view2=(normalized[0]*normalized[2]).sum(1).mean().item(),
                raw_feature_norms=[a.norm(dim=1).mean().item() for a in (h0,h1,h2)],
                raw_feature_variance=[a.var(dim=0,unbiased=False).mean().item() for a in (h0,h1,h2)],
                normalized_feature_variance=[a.var(dim=0,unbiased=False).mean().item() for a in normalized],
                encoder_grad_norm=encoder_grad,head_grad_norm=head_grad,
                contrastive_retrieval_accuracy=retrieval,
                positive_cosine=(scores[positives]*temperature).mean().item(),
                negative_cosine=(scores[negatives]*temperature).mean().item() if negatives.any() else 0.,
                candidates_per_anchor=len(scores)-1,positives_per_anchor=2,
                negatives_per_anchor=len(scores)-3,
                m_mean=m.mean().item(),m_std=m.std(unbiased=False).item(),m_min=m.min().item(),m_max=m.max().item(),
                m_view_means=m.mean(1).tolist(),w_mean=w.mean((0,1)).tolist(),
                w_std=w.std((0,1),unbiased=False).tolist(),w_view_means=w.mean(1).tolist(),
                current_batch=len(raw),paired_examples=len(raw_a),replay_batch=0 if ce_sample is None else len(ce_sample[0]),
                presentations=memory.seen,memory_size=memory.size,step_seconds=time.monotonic()-start)
        if debug:
            return row,dict(stored=stored,raw_a=raw_a,m=m,w=w)
        return row

    @staticmethod
    def update_infonce_classneg_cifar10(model,optimizer,memory,streams,raw,labels,device,debug=False,lambda_nce=1.,temperature=.1, *, contrastive_scores, infonce_loss, random_augmix_views, tensor):
        start = time.monotonic()
        ce_sample = memory.sample(streams.replay_ce) if memory.size else None
        logit_sample = memory.sample(streams.replay_logit) if memory.size else None
        raw_a = np.concatenate((raw,ce_sample[0])) if ce_sample is not None else raw
        labels_a = np.concatenate((labels,ce_sample[1])) if ce_sample is not None else labels
        y = torch.as_tensor(labels_a,dtype=torch.long,device=device)
        x0 = tensor(raw_a,device)
        views,m,w = random_augmix_views(raw_a,streams)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z,h = model(torch.cat((x0,views[0].to(device),views[1].to(device))),features=True)
        z0,z1,z2 = z.chunk(3)
        h0,h1,h2 = h.chunk(3)
        nce = infonce_loss(h0,h1,h2,temperature,y)
        current_ce = F.cross_entropy(z0[:len(raw)],y[:len(raw)])
        replay_ce = F.cross_entropy(z0[len(raw):],y[len(raw):]) if ce_sample is not None else z0.new_zeros(())
        replay_mse = z0.new_zeros(())
        if logit_sample is not None:
            replay_logits = model(tensor(logit_sample[0],device))
            replay_mse = F.mse_loss(replay_logits,logit_sample[2].to(device))
        stored = z0[:len(raw)].detach().cpu().clone()
        loss = current_ce+.5*replay_ce+.2*replay_mse+lambda_nce*nce
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite loss or gradients')
        encoder_grad = torch.stack([p.grad.square().sum() for p in model.net.parameters()]).sum().sqrt().item()
        head_grad = torch.stack([p.grad.square().sum() for p in model.head.parameters()]).sum().sqrt().item()
        optimizer.step()
        memory.add(raw,labels,stored,streams.insertion)
        with torch.no_grad():
            scores, positives = contrastive_scores(h0,h1,h2,temperature,y)
            negatives = torch.isfinite(scores) & ~positives
            negative_counts = negatives.sum(1)
            retrieval = positives[torch.arange(len(scores),device=device),scores.argmax(1)].float().mean().item()
            normalized = [F.normalize(a,p=2,dim=1,eps=1e-8) for a in (h0,h1,h2)]
            row = dict(current_ce=current_ce.item(),replay_ce=replay_ce.item(),replay_logit_mse=replay_mse.item(),
                infonce_loss=nce.item(),total_loss=loss.item(),
                current_clean_accuracy=(z0[:len(raw)].argmax(1)==y[:len(raw)]).float().mean().item(),
                clean_accuracy=(z0.argmax(1)==y).float().mean().item(),
                augmented_accuracy=((z1.argmax(1)==y).float().mean()+(z2.argmax(1)==y).float().mean()).item()/2,
                cosine_view1=(normalized[0]*normalized[1]).sum(1).mean().item(),
                cosine_view2=(normalized[0]*normalized[2]).sum(1).mean().item(),
                raw_feature_norms=[a.norm(dim=1).mean().item() for a in (h0,h1,h2)],
                raw_feature_variance=[a.var(dim=0,unbiased=False).mean().item() for a in (h0,h1,h2)],
                normalized_feature_variance=[a.var(dim=0,unbiased=False).mean().item() for a in normalized],
                encoder_grad_norm=encoder_grad,head_grad_norm=head_grad,
                contrastive_retrieval_accuracy=retrieval,
                positive_cosine=(scores[positives]*temperature).mean().item(),
                negative_cosine=(scores[negatives]*temperature).mean().item() if negatives.any() else 0.,
                candidates_per_anchor=negative_counts.float().mean().item()+2,positives_per_anchor=2,
                negatives_per_anchor=negative_counts.float().mean().item(),
                negatives_per_anchor_min=negative_counts.min().item(),
                negatives_per_anchor_max=negative_counts.max().item(),
                fraction_anchors_without_negatives=(negative_counts==0).float().mean().item(),
                m_mean=m.mean().item(),m_std=m.std(unbiased=False).item(),m_min=m.min().item(),m_max=m.max().item(),
                m_view_means=m.mean(1).tolist(),w_mean=w.mean((0,1)).tolist(),
                w_std=w.std((0,1),unbiased=False).tolist(),w_view_means=w.mean(1).tolist(),
                current_batch=len(raw),paired_examples=len(raw_a),replay_batch=0 if ce_sample is None else len(ce_sample[0]),
                presentations=memory.seen,memory_size=memory.size,step_seconds=time.monotonic()-start)
        if debug:
            return row,dict(stored=stored,raw_a=raw_a,m=m,w=w)
        return row

    @staticmethod
    def update_infonce_recon_cifar10(model,optimizer,memory,streams,raw,labels,device,debug=False,lambda_nce=1.,temperature=.1,lambda_recon=1., *, contrastive_scores, infonce_loss, random_augmix_views, tensor):
        start = time.monotonic()
        ce_sample = memory.sample(streams.replay_ce) if memory.size else None
        logit_sample = memory.sample(streams.replay_logit) if memory.size else None
        raw_a = np.concatenate((raw,ce_sample[0])) if ce_sample is not None else raw
        labels_a = np.concatenate((labels,ce_sample[1])) if ce_sample is not None else labels
        y = torch.as_tensor(labels_a,dtype=torch.long,device=device)
        x0 = tensor(raw_a,device)
        views,m,w = random_augmix_views(raw_a,streams)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.cat((x0,views[0].to(device),views[1].to(device)))
        z,h = model(inputs,features=True)
        reconstructed = model.decoder(h)
        recon = F.mse_loss(reconstructed, inputs)
        recon_views = [F.mse_loss(p,t).item() for p,t in zip(reconstructed.detach().chunk(3),inputs.chunk(3))]
        z0,z1,z2 = z.chunk(3)
        h0,h1,h2 = h.chunk(3)
        nce = infonce_loss(h0,h1,h2,temperature,y)
        current_ce = F.cross_entropy(z0[:len(raw)],y[:len(raw)])
        replay_ce = F.cross_entropy(z0[len(raw):],y[len(raw):]) if ce_sample is not None else z0.new_zeros(())
        replay_mse = z0.new_zeros(())
        if logit_sample is not None:
            replay_logits = model(tensor(logit_sample[0],device))
            replay_mse = F.mse_loss(replay_logits,logit_sample[2].to(device))
        stored = z0[:len(raw)].detach().cpu().clone()
        loss = current_ce+.5*replay_ce+.2*replay_mse+lambda_nce*nce+lambda_recon*recon
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite loss or gradients')
        encoder_grad = torch.stack([p.grad.square().sum() for p in model.net.parameters()]).sum().sqrt().item()
        head_grad = torch.stack([p.grad.square().sum() for p in model.head.parameters()]).sum().sqrt().item()
        decoder_grad = torch.stack([p.grad.square().sum() for p in model.decoder.parameters()]).sum().sqrt().item()
        optimizer.step()
        memory.add(raw,labels,stored,streams.insertion)
        with torch.no_grad():
            scores, positives = contrastive_scores(h0,h1,h2,temperature,y)
            negatives = torch.isfinite(scores) & ~positives
            negative_counts = negatives.sum(1)
            retrieval = positives[torch.arange(len(scores),device=device),scores.argmax(1)].float().mean().item()
            normalized = [F.normalize(a,p=2,dim=1,eps=1e-8) for a in (h0,h1,h2)]
            row = dict(current_ce=current_ce.item(),replay_ce=replay_ce.item(),replay_logit_mse=replay_mse.item(),
                infonce_loss=nce.item(),reconstruction_loss=recon.item(),reconstruction_view_losses=recon_views,decoder_grad_norm=decoder_grad,total_loss=loss.item(),
                current_clean_accuracy=(z0[:len(raw)].argmax(1)==y[:len(raw)]).float().mean().item(),
                clean_accuracy=(z0.argmax(1)==y).float().mean().item(),
                augmented_accuracy=((z1.argmax(1)==y).float().mean()+(z2.argmax(1)==y).float().mean()).item()/2,
                cosine_view1=(normalized[0]*normalized[1]).sum(1).mean().item(),
                cosine_view2=(normalized[0]*normalized[2]).sum(1).mean().item(),
                raw_feature_norms=[a.norm(dim=1).mean().item() for a in (h0,h1,h2)],
                raw_feature_variance=[a.var(dim=0,unbiased=False).mean().item() for a in (h0,h1,h2)],
                normalized_feature_variance=[a.var(dim=0,unbiased=False).mean().item() for a in normalized],
                encoder_grad_norm=encoder_grad,head_grad_norm=head_grad,
                contrastive_retrieval_accuracy=retrieval,
                positive_cosine=(scores[positives]*temperature).mean().item(),
                negative_cosine=(scores[negatives]*temperature).mean().item() if negatives.any() else 0.,
                candidates_per_anchor=negative_counts.float().mean().item()+2,positives_per_anchor=2,
                negatives_per_anchor=negative_counts.float().mean().item(),
                negatives_per_anchor_min=negative_counts.min().item(),
                negatives_per_anchor_max=negative_counts.max().item(),
                fraction_anchors_without_negatives=(negative_counts==0).float().mean().item(),
                m_mean=m.mean().item(),m_std=m.std(unbiased=False).item(),m_min=m.min().item(),m_max=m.max().item(),
                m_view_means=m.mean(1).tolist(),w_mean=w.mean((0,1)).tolist(),
                w_std=w.std((0,1),unbiased=False).tolist(),w_view_means=w.mean(1).tolist(),
                current_batch=len(raw),paired_examples=len(raw_a),replay_batch=0 if ce_sample is None else len(ce_sample[0]),
                presentations=memory.seen,memory_size=memory.size,step_seconds=time.monotonic()-start)
        if debug:
            return row,dict(stored=stored,raw_a=raw_a,m=m,w=w,inputs=inputs.detach(),reconstructed=reconstructed.detach())
        return row

    @staticmethod
    def update_mse_infonce_cifar10(model,optimizer,memory,streams,raw,labels,device,debug=False,lambda_nce=.5,temperature=.1,lambda_mse=.5, *, contrastive_scores, feature_mse_loss, infonce_loss, random_augmix_views, tensor):
        start = time.monotonic()
        ce_sample = memory.sample(streams.replay_ce) if memory.size else None
        logit_sample = memory.sample(streams.replay_logit) if memory.size else None
        raw_a = np.concatenate((raw,ce_sample[0])) if ce_sample is not None else raw
        labels_a = np.concatenate((labels,ce_sample[1])) if ce_sample is not None else labels
        y = torch.as_tensor(labels_a,dtype=torch.long,device=device)
        x0 = tensor(raw_a,device)
        views,m,w = random_augmix_views(raw_a,streams)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z,h = model(torch.cat((x0,views[0].to(device),views[1].to(device))),features=True)
        z0,z1,z2 = z.chunk(3)
        h0,h1,h2 = h.chunk(3)
        nce = infonce_loss(h0,h1,h2,temperature,y)
        mse = feature_mse_loss(h0,h1,h2)
        current_ce = F.cross_entropy(z0[:len(raw)],y[:len(raw)])
        replay_ce = F.cross_entropy(z0[len(raw):],y[len(raw):]) if ce_sample is not None else z0.new_zeros(())
        replay_mse = z0.new_zeros(())
        if logit_sample is not None:
            replay_logits = model(tensor(logit_sample[0],device))
            replay_mse = F.mse_loss(replay_logits,logit_sample[2].to(device))
        stored = z0[:len(raw)].detach().cpu().clone()
        # Diagnostic VJPs neither accumulate parameter .grad nor change RNG/BN state.
        encoder_params = tuple(model.net.parameters())
        mse_grads = torch.autograd.grad(mse,encoder_params,retain_graph=True)
        nce_grads = torch.autograd.grad(nce,encoder_params,retain_graph=True)
        mse_grad_norm = torch.stack([g.square().sum() for g in mse_grads]).sum().sqrt()
        nce_grad_norm = torch.stack([g.square().sum() for g in nce_grads]).sum().sqrt()
        grad_dot = torch.stack([(a*b).sum() for a,b in zip(mse_grads,nce_grads)]).sum()
        grad_cosine = grad_dot/(mse_grad_norm*nce_grad_norm).clamp_min(1e-12)
        loss = current_ce+.5*replay_ce+.2*replay_mse+lambda_nce*nce+lambda_mse*mse
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite loss or gradients')
        encoder_grad = torch.stack([p.grad.square().sum() for p in model.net.parameters()]).sum().sqrt().item()
        head_grad = torch.stack([p.grad.square().sum() for p in model.head.parameters()]).sum().sqrt().item()
        optimizer.step()
        memory.add(raw,labels,stored,streams.insertion)
        with torch.no_grad():
            scores, positives = contrastive_scores(h0,h1,h2,temperature,y)
            negatives = torch.isfinite(scores) & ~positives
            negative_counts = negatives.sum(1)
            retrieval = positives[torch.arange(len(scores),device=device),scores.argmax(1)].float().mean().item()
            normalized = [F.normalize(a,p=2,dim=1,eps=1e-8) for a in (h0,h1,h2)]
            row = dict(current_ce=current_ce.item(),replay_ce=replay_ce.item(),replay_logit_mse=replay_mse.item(),
                infonce_loss=nce.item(),feature_mse_loss=mse.item(),
                weighted_nce=(lambda_nce*nce).item(),weighted_mse=(lambda_mse*mse).item(),
                mse_encoder_grad_norm=mse_grad_norm.item(),nce_encoder_grad_norm=nce_grad_norm.item(),
                weighted_mse_encoder_grad_norm=lambda_mse*mse_grad_norm.item(),
                weighted_nce_encoder_grad_norm=lambda_nce*nce_grad_norm.item(),
                consistency_grad_cosine=grad_cosine.item(),total_loss=loss.item(),
                current_clean_accuracy=(z0[:len(raw)].argmax(1)==y[:len(raw)]).float().mean().item(),
                clean_accuracy=(z0.argmax(1)==y).float().mean().item(),
                augmented_accuracy=((z1.argmax(1)==y).float().mean()+(z2.argmax(1)==y).float().mean()).item()/2,
                cosine_view1=(normalized[0]*normalized[1]).sum(1).mean().item(),
                cosine_view2=(normalized[0]*normalized[2]).sum(1).mean().item(),
                raw_feature_norms=[a.norm(dim=1).mean().item() for a in (h0,h1,h2)],
                raw_feature_variance=[a.var(dim=0,unbiased=False).mean().item() for a in (h0,h1,h2)],
                normalized_feature_variance=[a.var(dim=0,unbiased=False).mean().item() for a in normalized],
                encoder_grad_norm=encoder_grad,head_grad_norm=head_grad,
                contrastive_retrieval_accuracy=retrieval,
                positive_cosine=(scores[positives]*temperature).mean().item(),
                negative_cosine=(scores[negatives]*temperature).mean().item() if negatives.any() else 0.,
                candidates_per_anchor=negative_counts.float().mean().item()+2,positives_per_anchor=2,
                negatives_per_anchor=negative_counts.float().mean().item(),
                negatives_per_anchor_min=negative_counts.min().item(),
                negatives_per_anchor_max=negative_counts.max().item(),
                fraction_anchors_without_negatives=(negative_counts==0).float().mean().item(),
                m_mean=m.mean().item(),m_std=m.std(unbiased=False).item(),m_min=m.min().item(),m_max=m.max().item(),
                m_view_means=m.mean(1).tolist(),w_mean=w.mean((0,1)).tolist(),
                w_std=w.std((0,1),unbiased=False).tolist(),w_view_means=w.mean(1).tolist(),
                current_batch=len(raw),paired_examples=len(raw_a),replay_batch=0 if ce_sample is None else len(ce_sample[0]),
                presentations=memory.seen,memory_size=memory.size,step_seconds=time.monotonic()-start)
        if debug:
            return row,dict(stored=stored,raw_a=raw_a,m=m,w=w)
        return row

    @staticmethod
    def update_multihead_cifar10(model,optimizer,memory,streams,raw,labels,device,debug=False,lambda_aug=1.,lambda_div=.1, *, diversity_loss, ensemble, head_ce, random_augmix_views, tensor):
        start = time.monotonic()
        ce_sample = memory.sample(streams.replay_ce) if memory.size else None
        logit_sample = memory.sample(streams.replay_logit) if memory.size else None
        raw_a = np.concatenate((raw,ce_sample[0])) if ce_sample is not None else raw
        labels_a = np.concatenate((labels,ce_sample[1])) if ce_sample is not None else labels
        y = torch.as_tensor(labels_a,dtype=torch.long,device=device)
        x0 = tensor(raw_a,device)
        views,m,w = random_augmix_views(raw_a,streams)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        joint_inputs = torch.cat((x0,views[0].to(device),views[1].to(device)))
        z,h = model(joint_inputs,features=True)
        z0,z1,z2 = z.chunk(3)
        h0,h1,h2 = h.chunk(3)
        augmented_ce = .5*(head_ce(z1,y)+head_ce(z2,y))
        diversity = diversity_loss(h)
        current_ce = head_ce(z0[:len(raw)],y[:len(raw)])
        replay_ce = head_ce(z0[len(raw):],y[len(raw):]) if ce_sample is not None else z0.new_zeros(())
        replay_mse = z0.new_zeros(())
        if logit_sample is not None:
            replay_logits = model(tensor(logit_sample[0],device))
            replay_mse = F.mse_loss(replay_logits,logit_sample[2].to(device))
        stored = z0[:len(raw)].detach().cpu().clone()
        loss = current_ce+.5*replay_ce+.2*replay_mse+lambda_aug*augmented_ce+lambda_div*diversity
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite loss or gradients')
        encoder_grad = torch.stack([p.grad.square().sum() for p in model.net.parameters()]).sum().sqrt().item()
        head_grad = torch.stack([p.grad.square().sum() for p in model.head.parameters()]).sum().sqrt().item()
        optimizer.step()
        memory.add(raw,labels,stored,streams.insertion)
        with torch.no_grad():
            normalized = [F.normalize(a,p=2,dim=-1,eps=1e-8) for a in (h0,h1,h2)]
            row = dict(current_ce=current_ce.item(),replay_ce=replay_ce.item(),replay_logit_mse=replay_mse.item(),
                augmented_ce=augmented_ce.item(),diversity_loss=diversity.item(),
                weighted_augmented_ce=(lambda_aug*augmented_ce).item(),
                weighted_diversity=(lambda_div*diversity).item(),
                total_loss=loss.item(),
                current_clean_accuracy=(ensemble(z0[:len(raw)]).argmax(1)==y[:len(raw)]).float().mean().item(),
                clean_accuracy=(ensemble(z0).argmax(1)==y).float().mean().item(),
                augmented_accuracy=((ensemble(z1).argmax(1)==y).float().mean()+(ensemble(z2).argmax(1)==y).float().mean()).item()/2,
                cosine_view1=(normalized[0]*normalized[1]).sum(-1).mean().item(),
                cosine_view2=(normalized[0]*normalized[2]).sum(-1).mean().item(),
                raw_feature_norms=[a.norm(dim=-1).mean().item() for a in (h0,h1,h2)],
                raw_feature_variance=[a.var(dim=0,unbiased=False).mean().item() for a in (h0,h1,h2)],
                normalized_feature_variance=[a.var(dim=0,unbiased=False).mean().item() for a in normalized],
                encoder_grad_norm=encoder_grad,head_grad_norm=head_grad,
                m_mean=m.mean().item(),m_std=m.std(unbiased=False).item(),m_min=m.min().item(),m_max=m.max().item(),
                m_view_means=m.mean(1).tolist(),w_mean=w.mean((0,1)).tolist(),
                w_std=w.std((0,1),unbiased=False).tolist(),w_view_means=w.mean(1).tolist(),
                current_batch=len(raw),paired_examples=len(raw_a),replay_batch=0 if ce_sample is None else len(ce_sample[0]),
                presentations=memory.seen,memory_size=memory.size,step_seconds=time.monotonic()-start)
        row.update(head_clean_accuracy=(z0.argmax(-1)==y[:,None]).float().mean(0).tolist(),
            head_feature_norms=h0.norm(dim=-1).mean(0).tolist(),
            zero_branch_fraction=(h0.norm(dim=-1)<1e-8).float().mean().item())
        if debug:
            return row,dict(stored=stored,raw_a=raw_a,m=m,w=w)
        return row

class BiLevelTrainer:
    """Own classifier/controller resources and advance recurrent state per batch.

    State remains detached by the selected algorithm. Assign ``state`` when
    restoring a checkpoint; optimizers and replay memory retain their existing
    state_dict protocols. Random-mixer controls use the same interface.
    """
    def __init__(self, model, controller, optimizer, mixer_optimizer, memory,
                 state, device, *, update, **options):
        self.model, self.controller = model, controller
        self.optimizer, self.mixer_optimizer = optimizer, mixer_optimizer
        self.memory, self.state, self.device = memory, state, device
        self.update, self.options = update, options

    def step(self, raw, labels, **options):
        result = self.update(self.model, self.controller, self.optimizer,
                             self.mixer_optimizer, self.memory, self.state,
                             raw, labels, self.device, **(self.options | options))
        self.state = result[0]
        return result[1] if len(result) == 2 else result[1:]

    @staticmethod
    def update_chain_aware_mixing(model,controller,optimizer,mixer_optimizer,memory,state,raw,labels,device,j=3,debug=False,mixer='learned',kind='infonce',temperature=.1, *, outer_kind='js', active_slots, candidate_descriptors, chains, clone_buffers, consistency_loss, context_for, forward, mixed_views, ops, outer_objective, tensor):
        if mixer!='learned' or kind not in ('infonce','js'):
            raise ValueError('This variant requires a learned mixer and seen InfoNCE or JS')
        if outer_kind not in ('infonce','js','feature_match','seen_js_feature_match'):
            raise ValueError('Outer objective must be InfoNCE, JS, feature_match or seen_js_feature_match')
        x=tensor(raw,device);y=torch.as_tensor(labels,device=device)
        ce_raw=memory.sample(32,device,raw=True) if memory.size else None
        replay=(tensor(ce_raw[0],device),ce_raw[1],ce_raw[2]) if ce_raw is not None else None
        distill=memory.sample(32,device) if memory.size else None
        raw_a=np.concatenate((raw,ce_raw[0])) if ce_raw is not None else raw
        xa=tensor(raw_a,device);ya=torch.cat((y,ce_raw[1])) if ce_raw is not None else y
        b,r=len(raw),0 if ce_raw is None else len(ce_raw[0])
        slots=active_slots(b,r,device)
        context=context_for(model,x,y,replay)
        incoming=tuple(v[slots].detach() for v in state)
        next_state=tuple(v.detach().clone() for v in state)
        perm=np.random.permutation(9);q=int(np.random.randint(5,8))
        seen=chains(raw_a,[ops.augmentations[i] for i in perm[:q]],device)
        # Preserve the previous runner's seen/unseen chain sampling order.
        unseen=chains(raw_a,[ops.augmentations[i] for i in perm[q:]],device)
        seen_candidates=candidate_descriptors(model,xa,seen)
        with torch.no_grad():
            m,w,advanced=controller(context,incoming,seen_candidates)
            for target,value in zip(next_state,advanced):target[slots]=value
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
        if outer_kind in ('feature_match', 'seen_js_feature_match'):
            # Score seen candidates with the updated classifier. Unseen chains
            # are independent, unmixed targets and never enter the controller.
            outer_branches = seen
            outer_candidates = candidate_descriptors(model, xa, seen)
            with torch.no_grad():
                target_images = torch.cat([chain for view in unseen for chain in view])
                target_features = forward(model, target_images, params, buffers)[1]
                target_features = target_features.reshape(2, 3, len(xa), -1).detach()
            outer_options = dict(target_features=target_features)
        else:
            outer_branches = unseen
            outer_candidates = candidate_descriptors(model, xa, unseen)
            outer_options = {}
        outer_rows=[]
        for _ in range(j):
            mixer_optimizer.zero_grad(set_to_none=True)
            outer_loss,outer_penalty,om,ow,outer_ce=outer_objective(model,controller,context,incoming,xa,outer_branches,params,buffers,ya,outer_kind,temperature,candidates=outer_candidates,**outer_options)
            outer_loss.backward()
            grads=[p.grad for p in controller.parameters() if p.grad is not None]
            if not torch.isfinite(outer_loss) or not grads or not all(torch.isfinite(g).all() for g in grads):raise RuntimeError('Nonfinite mixer loss/gradient')
            grad_norm=torch.stack([g.square().sum() for g in grads]).sum().sqrt().item()
            outer_rows.append(dict(m=om.mean().item(),m_per_example=om.detach().cpu().tolist(),w=ow.detach().cpu().tolist(),consistency_loss=outer_penalty.item(),grad_norm=grad_norm))
            if outer_kind == 'seen_js_feature_match':
                outer_rows[-1]['mixer_loss'] = outer_loss.item()
            mixer_optimizer.step()
        with torch.no_grad():
            after_total,after_penalty,after_m,after_w,_=outer_objective(model,controller,context,incoming,xa,outer_branches,params,buffers,ya,outer_kind,temperature,candidates=outer_candidates,**outer_options)
        memory.add(raw,labels,old_logits)
        row=dict(m_before=m.mean().item(),m_after=after_m.mean().item(),
            m_before_per_example=m.flatten().cpu().tolist(),m_after_per_example=after_m.flatten().cpu().tolist(),
            w_before=w.cpu().tolist(),w_after=after_w.cpu().tolist(),mixer=mixer,consistency=kind,outer_consistency=outer_kind,fixed_m=.5,
            weight_entropy_before=(-(w*w.clamp_min(1e-8).log()).sum(-1)).mean().item(),
            weight_entropy_after=(-(after_w*after_w.clamp_min(1e-8).log()).sum(-1)).mean().item(),
            weight_max_after=after_w.max(-1).values.mean().item(),
            descriptor_feature_rms=seen_candidates[...,:model.feature_dimension].square().mean().sqrt().item(),
            descriptor_prediction_rms=seen_candidates[...,model.feature_dimension:].square().mean().sqrt().item(),
            sampling_sha256=digest.hexdigest(),seen_consistency=penalty.item(),outer_consistency_after=after_penalty.item(),
            current_ce=ce.item(),replay_ce=replay_ce.item(),replay_mse=replay_mse.item(),classifier_loss=loss.item(),
            outer_ce=outer_ce.item(),mixer_updates=outer_rows,q=q,batch_size=b,replay_batch=r,augmented_batch=b+r,
            memory_size=memory.size,presentations=memory.seen)
        if outer_kind in ('feature_match', 'seen_js_feature_match'):
            row['feature_match_after'] = after_penalty.item()
            if outer_kind == 'seen_js_feature_match':
                row['mixer_loss_after'] = after_total.item()
            for update in outer_rows:
                update['feature_match'] = update['consistency_loss']
        if debug:return next_state,row,dict(stored=old_logits,m=m,w=w,context=context,seen_candidates=seen_candidates)
        return next_state,row

    @staticmethod
    def update_matched_infonce_js(model,controller,optimizer,mixer_optimizer,memory,state,raw,labels,device,j=3,debug=False,mixer='learned',kind='infonce',temperature=.1,outer_kind='js', *, active_slots, chains, clone_buffers, consistency_loss, context_for, forward, mixed_views, ops, outer_objective, random_weights, tensor):
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

    @staticmethod
    def update_matched_mixing(model,controller,optimizer,mixer_optimizer,memory,state,raw,labels,device,j=3,debug=False,mixer='learned',kind='js',temperature=.1, *, active_slots, chains, clone_buffers, consistency_loss, context_for, forward, mixed_views, ops, outer_objective, random_weights, tensor):
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
                outer_loss,outer_penalty,om,ow,outer_ce=outer_objective(model,controller,context,incoming,xa,unseen,params,buffers,ya,kind,temperature)
                outer_loss.backward()
                grads=[p.grad for p in controller.parameters() if p.grad is not None]
                if not torch.isfinite(outer_loss) or not grads or not all(torch.isfinite(g).all() for g in grads):raise RuntimeError('Nonfinite mixer loss/gradient')
                grad_norm=torch.stack([g.square().sum() for g in grads]).sum().sqrt().item()
                outer_rows.append(dict(m=om.mean().item(),m_per_example=om.detach().cpu().tolist(),w=ow.detach().cpu().tolist(),consistency_loss=outer_penalty.item(),grad_norm=grad_norm))
                mixer_optimizer.step()
            with torch.no_grad():
                _,after_penalty,after_m,after_w,_=outer_objective(model,controller,context,incoming,xa,unseen,params,buffers,ya,kind,temperature)
        else:
            with torch.no_grad():
                uz,uh=forward(model,torch.cat((xa,*mixed_views(xa,unseen,m,w))),params,buffers)
                after_penalty=consistency_loss(uz.chunk(3),uh.chunk(3),ya,kind,temperature)
                outer_ce=F.cross_entropy(uz.chunk(3)[0],ya)
            after_m,after_w=m,w
        memory.add(raw,labels,old_logits)
        row=dict(m_before=m.mean().item(),m_after=after_m.mean().item(),
            m_before_per_example=m.flatten().cpu().tolist(),m_after_per_example=after_m.flatten().cpu().tolist(),
            w_before=w.cpu().tolist(),w_after=after_w.cpu().tolist(),mixer=mixer,consistency=kind,
            sampling_sha256=digest.hexdigest(),seen_consistency=penalty.item(),outer_consistency_after=after_penalty.item(),
            current_ce=ce.item(),replay_ce=replay_ce.item(),replay_mse=replay_mse.item(),classifier_loss=loss.item(),
            outer_ce=outer_ce.item(),mixer_updates=outer_rows,q=q,batch_size=b,replay_batch=r,augmented_batch=b+r,
            memory_size=memory.size,presentations=memory.seen)
        if debug:return next_state,row,dict(stored=old_logits,m=m,w=w,context=context)
        return next_state,row

    @staticmethod
    def update_metamix_jointbn(model, controller, optimizer, mixer_optimizer, memory, state, raw, labels, device, j=3,debug=False, *, active_slots, blend, chains, clone_buffers, context_for, forward, js, ops, outer_objective, tensor):
        x = tensor(raw,device)
        y = torch.as_tensor(labels,device=device)
        ce_raw = memory.sample(32,device,raw=True) if memory.size else None
        replay = (tensor(ce_raw[0],device),ce_raw[1],ce_raw[2]) if ce_raw is not None else None
        distill = memory.sample(32,device) if memory.size else None
        raw_a = np.concatenate((raw,ce_raw[0])) if ce_raw is not None else raw
        xa = tensor(raw_a,device)
        ya = torch.cat((y,ce_raw[1])) if ce_raw is not None else y
        b,r = len(raw),0 if ce_raw is None else len(ce_raw[0])
        slots = active_slots(b,r,device)
        context = context_for(model,x,y,replay)
        incoming = tuple(v[slots].detach() for v in state)
        # Commit Algorithm 3's pre-solver temporal transition exactly once.
        with torch.no_grad():
            m,w,advanced = controller(context,incoming)
            next_state = tuple(v.detach().clone() for v in state)
            for target,value in zip(next_state,advanced): target[slots] = value
        perm = np.random.permutation(9)
        q = int(np.random.randint(5,8))
        seen = chains(raw_a,[ops.augmentations[i] for i in perm[:q]],device)
        unseen = chains(raw_a,[ops.augmentations[i] for i in perm[q:]],device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        joined=torch.cat((xa,*(blend(xa,v,m,w) for v in seen)))
        joint_logits,_ = forward(model,joined)
        logits,seen1,seen2=joint_logits.chunk(3)
        old_logits = logits[:b].detach().clone()
        ce = F.cross_entropy(logits[:b],y)
        replay_ce = F.cross_entropy(logits[b:],replay[1]) if r else ce.new_zeros(())
        replay_mse = F.mse_loss(forward(model,distill[0])[0],distill[2]) if distill else ce.new_zeros(())
        seen_logits = [seen1,seen2]
        seen_js = js(logits,*seen_logits)
        loss = ce+.5*replay_ce+.2*replay_mse+seen_js
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite classifier loss/gradient')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        params = {k:v.detach() for k,v in model.named_parameters()}
        buffers = clone_buffers(model)
        outer_rows=[]
        for _ in range(j):
            mixer_optimizer.zero_grad(set_to_none=True)
            outer_loss,outer_js,om,ow,outer_ce = outer_objective(model,controller,context,incoming,xa,unseen,params,buffers,ya)
            outer_loss.backward()
            grads=[p.grad for p in controller.parameters() if p.grad is not None]
            if not torch.isfinite(outer_loss) or not grads or not all(torch.isfinite(g).all() for g in grads):
                raise RuntimeError('Nonfinite/missing mixer gradients')
            grad_norm=torch.stack([g.square().sum() for g in grads]).sum().sqrt().item()
            outer_rows.append(dict(m=om.mean().item(),m_per_example=om.detach().cpu().tolist(),w=ow.detach().cpu().tolist(),js=outer_js.item(),grad_norm=grad_norm))
            mixer_optimizer.step()
        with torch.no_grad():
            _,after_js,after_m,after_w,_=outer_objective(model,controller,context,incoming,xa,unseen,params,buffers,ya)
            # This diagnostic recomputation must NOT overwrite next_state.
            seen_current=js(logits[:b],*(z[:b] for z in seen_logits)).item()
            seen_replay=js(logits[b:],*(z[b:] for z in seen_logits)).item() if r else None
        memory.add(raw,labels,old_logits)
        row=dict(m_before=m.mean().item(),m_after=after_m.mean().item(),
            m_before_per_example=m.cpu().tolist(),m_after_per_example=after_m.cpu().tolist(),
            w_before=w.cpu().tolist(),w_after=after_w.cpu().tolist(),
            m_current_mean=after_m[:b].mean().item(),m_replay_mean=after_m[b:].mean().item() if r else None,
            seen_js=seen_js.item(),seen_js_current=seen_current,seen_js_replay=seen_replay,
            outer_js=outer_rows[-1]['js'],outer_js_after=after_js.item(),
            current_ce=ce.item(),replay_ce=replay_ce.item(),replay_mse=replay_mse.item(),
            classifier_loss=loss.item(),outer_ce=outer_ce.item(),mixer_updates=outer_rows,
            q=q,batch_size=b,replay_batch=r,augmented_batch=b+r,memory_size=memory.size,presentations=memory.seen)
        if debug:
            return next_state,row,dict(context=context,slots=slots,incoming=incoming,advanced=advanced,
                raw_a=raw_a,stored=old_logits,xa=xa,unseen=unseen,params=params,buffers=buffers,
                outer_ce=outer_ce,labels_a=ya)
        return next_state,row

    @staticmethod
    def update_metamix_long(model, controller, optimizer, mixer_optimizer, memory, state, raw, labels, device, j=3, *, blend, chains, clone_buffers, context_for, forward, js, ops, tensor):
        x = tensor(raw, device)
        y = torch.as_tensor(labels, device=device)
        replay = memory.sample(32, device) if memory.size else None
        distill = memory.sample(32, device) if memory.size else None
        context = context_for(model, x, y, replay)
        incoming = tuple(s.detach() for s in state)
        with torch.no_grad():
            m, w, _ = controller(context, incoming)
        before_m, before_w = m.item(), w.cpu().tolist()
        perm = np.random.permutation(9)
        q = int(np.random.randint(5, 8))
        seen_pool = [ops.augmentations[i] for i in perm[:q]]
        unseen_pool = [ops.augmentations[i] for i in perm[q:]]
        seen = chains(raw, seen_pool, device)
        unseen = chains(raw, unseen_pool, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = forward(model, x)
        old_logits = logits.detach().clone()
        ce = F.cross_entropy(logits, y)
        zero = ce.new_zeros(())
        replay_ce = F.cross_entropy(forward(model, replay[0])[0], replay[1]) if replay else zero
        replay_mse = F.mse_loss(forward(model, distill[0])[0], distill[2]) if distill else zero
        seen_js = js(logits, *(forward(model, blend(x, branch, m, w))[0] for branch in seen))
        loss = ce + .5*replay_ce + .2*replay_mse + seen_js
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite classifier gradients')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # Updated classifier is frozen; gradients propagate only through blending/controller.
        model.eval()
        params = {k: v.detach() for k, v in model.named_parameters()}
        buffers = clone_buffers(model)
        with torch.no_grad():
            outer_logits = forward(model, x, params, buffers)[0]
            outer_ce = F.cross_entropy(outer_logits, y)
        outer_rows = []
        for _ in range(j):
            mixer_optimizer.zero_grad(set_to_none=True)
            om, ow, _ = controller(context, incoming)
            outer_js = js(outer_logits, *(forward(model, blend(x, b, om, ow), params, buffers)[0] for b in unseen))
            outer_loss = outer_ce + outer_js
            outer_loss.backward()
            grads = [p.grad for p in controller.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise RuntimeError('Missing/nonfinite mixer gradients')
            grad_norm = torch.stack([g.square().sum() for g in grads]).sum().sqrt().item()
            outer_rows.append(dict(m=om.item(), w=ow.detach().cpu().tolist(),
                                   js=outer_js.item(), grad_norm=grad_norm))
            mixer_optimizer.step()
        with torch.no_grad():
            after_m, after_w, state = controller(context, incoming)
            # Evaluate the same pseudo-unseen chains after the final mixer update.
            after_js = js(outer_logits, *(forward(model, blend(x, b, after_m, after_w), params, buffers)[0] for b in unseen))
        memory.add(raw, labels, old_logits)
        row = dict(m_before=before_m, m_after=after_m.item(), w_before=before_w,
                   w_after=after_w.cpu().tolist(), seen_js=seen_js.item(),
                   outer_js=outer_rows[-1]['js'], outer_js_after=after_js.item(),
                   current_ce=ce.item(), replay_ce=replay_ce.item(), replay_mse=replay_mse.item(),
                   classifier_loss=loss.item(), outer_ce=outer_ce.item(), mixer_updates=outer_rows,
                   q=q, batch_size=len(raw), memory_size=memory.size, presentations=memory.seen)
        return tuple(s.detach() for s in state), row

    @staticmethod
    def update_metamix_revised(model, controller, optimizer, mixer_optimizer, memory, state, raw, labels, device, j=3,debug=False, *, active_slots, blend, chains, clone_buffers, context_for, forward, js, ops, outer_objective, tensor):
        x = tensor(raw,device)
        y = torch.as_tensor(labels,device=device)
        ce_raw = memory.sample(32,device,raw=True) if memory.size else None
        replay = (tensor(ce_raw[0],device),ce_raw[1],ce_raw[2]) if ce_raw is not None else None
        distill = memory.sample(32,device) if memory.size else None
        raw_a = np.concatenate((raw,ce_raw[0])) if ce_raw is not None else raw
        xa = tensor(raw_a,device)
        ya = torch.cat((y,ce_raw[1])) if ce_raw is not None else y
        b,r = len(raw),0 if ce_raw is None else len(ce_raw[0])
        slots = active_slots(b,r,device)
        context = context_for(model,x,y,replay)
        incoming = tuple(v[slots].detach() for v in state)
        # Commit Algorithm 3's pre-solver temporal transition exactly once.
        with torch.no_grad():
            m,w,advanced = controller(context,incoming)
            next_state = tuple(v.detach().clone() for v in state)
            for target,value in zip(next_state,advanced): target[slots] = value
        perm = np.random.permutation(9)
        q = int(np.random.randint(5,8))
        seen = chains(raw_a,[ops.augmentations[i] for i in perm[:q]],device)
        unseen = chains(raw_a,[ops.augmentations[i] for i in perm[q:]],device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits,_ = forward(model,xa)
        old_logits = logits[:b].detach().clone()
        ce = F.cross_entropy(logits[:b],y)
        replay_ce = F.cross_entropy(logits[b:],replay[1]) if r else ce.new_zeros(())
        replay_mse = F.mse_loss(forward(model,distill[0])[0],distill[2]) if distill else ce.new_zeros(())
        seen_logits = [forward(model,blend(xa,v,m,w))[0] for v in seen]
        seen_js = js(logits,*seen_logits)
        loss = ce+.5*replay_ce+.2*replay_mse+seen_js
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite classifier loss/gradient')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        params = {k:v.detach() for k,v in model.named_parameters()}
        buffers = clone_buffers(model)
        with torch.no_grad():
            outer_logits = forward(model,xa,params,buffers)[0]
            outer_ce = F.cross_entropy(outer_logits,ya)
        outer_rows=[]
        for _ in range(j):
            mixer_optimizer.zero_grad(set_to_none=True)
            outer_loss,outer_js,om,ow = outer_objective(model,controller,context,incoming,xa,unseen,params,buffers,outer_ce,outer_logits)
            outer_loss.backward()
            grads=[p.grad for p in controller.parameters() if p.grad is not None]
            if not torch.isfinite(outer_loss) or not grads or not all(torch.isfinite(g).all() for g in grads):
                raise RuntimeError('Nonfinite/missing mixer gradients')
            grad_norm=torch.stack([g.square().sum() for g in grads]).sum().sqrt().item()
            outer_rows.append(dict(m=om.mean().item(),m_per_example=om.detach().cpu().tolist(),w=ow.detach().cpu().tolist(),js=outer_js.item(),grad_norm=grad_norm))
            mixer_optimizer.step()
        with torch.no_grad():
            _,after_js,after_m,after_w=outer_objective(model,controller,context,incoming,xa,unseen,params,buffers,outer_ce,outer_logits)
            # This diagnostic recomputation must NOT overwrite next_state.
            seen_current=js(logits[:b],*(z[:b] for z in seen_logits)).item()
            seen_replay=js(logits[b:],*(z[b:] for z in seen_logits)).item() if r else None
        memory.add(raw,labels,old_logits)
        row=dict(m_before=m.mean().item(),m_after=after_m.mean().item(),
            m_before_per_example=m.cpu().tolist(),m_after_per_example=after_m.cpu().tolist(),
            w_before=w.cpu().tolist(),w_after=after_w.cpu().tolist(),
            m_current_mean=after_m[:b].mean().item(),m_replay_mean=after_m[b:].mean().item() if r else None,
            seen_js=seen_js.item(),seen_js_current=seen_current,seen_js_replay=seen_replay,
            outer_js=outer_rows[-1]['js'],outer_js_after=after_js.item(),
            current_ce=ce.item(),replay_ce=replay_ce.item(),replay_mse=replay_mse.item(),
            classifier_loss=loss.item(),outer_ce=outer_ce.item(),mixer_updates=outer_rows,
            q=q,batch_size=b,replay_batch=r,augmented_batch=b+r,memory_size=memory.size,presentations=memory.seen)
        if debug:
            return next_state,row,dict(context=context,slots=slots,incoming=incoming,advanced=advanced,
                raw_a=raw_a,stored=old_logits,xa=xa,unseen=unseen,params=params,buffers=buffers,
                outer_ce=outer_ce,outer_logits=outer_logits)
        return next_state,row
