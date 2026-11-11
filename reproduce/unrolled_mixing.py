"""One-step differentiable SGD for the chain-aware and MetaMix experiments."""
from functools import partial

import numpy as np
import torch
from torch.nn import functional as F

import chain_aware_mixing as base


def step(model, controller, optimizer, mixer_optimizer, memory, state, raw, labels,
         device, *, variant, kind, temperature, outer_kind, j, mixer, severity_upperbound=3, fixed_m=.5):
    if outer_kind != 'js' or j < 1 or mixer != 'learned':
        raise ValueError('The unrolled experiment requires a learned mixer and outer JS.')
    chain_aware = variant == 'chain_random'
    if variant not in ('chain_random', 'metamix', 'metamix_fixed_m') or kind not in ('js', 'infonce'):
        raise ValueError(f'Unsupported mixer or classifier consistency: {variant}, {kind}')
    group, = optimizer.param_groups
    if type(optimizer) is not torch.optim.SGD or any(
        group[key] for key in ('momentum', 'weight_decay', 'maximize')
    ):
        raise ValueError('The virtual update implements plain SGD.')
    x = base.tensor(raw, device)
    y = torch.as_tensor(labels, device=device)
    replay_raw = memory.sample(32, device, raw=True) if memory.size else None
    replay = (base.tensor(replay_raw[0], device), *replay_raw[1:]) if replay_raw else None
    distill = memory.sample(32, device) if replay else None
    images = np.concatenate((raw, replay_raw[0])) if replay else raw
    xa = base.tensor(images, device)
    ya = torch.cat((y, replay[1])) if replay else y
    b = len(x)
    slots = base.active_slots(b, len(xa) - b, device, len(state[0]) - 32)
    incoming = tuple(v[slots].detach() for v in state)
    context = base.context_for(model, x, y, replay)
    permutation = np.random.permutation(len(base.ops.augmentations))
    cut = np.random.randint(5, 8)
    def sample(indices):
        return base.chains(images, [base.ops.augmentations[i] for i in indices], device, severity_upperbound=severity_upperbound)
    seen, unseen = sample(permutation[:cut]), sample(permutation[cut:])
    if chain_aware:
        describe = base.candidate_descriptors(model, xa, seen)
        def score():
            original, weights, advanced = controller(context, incoming, describe)
            return torch.full_like(original, fixed_m), weights, advanced
        weights = xa.new_tensor(base.MIXTURE_RNG.dirichlet(np.ones(3), (2, len(xa))))
        fixed_views = base.mixed_views(xa, unseen, xa.new_full((2, len(xa)), fixed_m), weights)
    elif variant == 'metamix_fixed_m':
        def score():
            learned_m, weights, advanced = controller(context, incoming)
            return torch.full_like(learned_m, fixed_m), weights, advanced
    else:
        score = partial(controller, context, incoming)
    params = dict(model.named_parameters())
    # Hold evaluation statistics fixed throughout the local optimization.
    query_buffers = base.clone_buffers(model)
    next_state = tuple(v.detach().clone() for v in state)
    updates = []
    for iteration in range(j):
        m, w, advanced = score()
        buffers = base.clone_buffers(model)
        model.train()
        z, h = base.forward(model, torch.cat([xa, *base.mixed_views(xa, seen, m, w)]),
                            params, buffers)
        logits = z.chunk(3)
        penalty = base.consistency_loss(logits, h.chunk(3), ya, kind, temperature)
        support = F.cross_entropy(logits[0][:b], y) + penalty
        if replay:
            support = support + .5 * F.cross_entropy(logits[0][b:], replay[1])
            rz, _ = base.forward(model, distill[0], params, buffers)
            support = support + .2 * F.mse_loss(rz, distill[2])
        grads = torch.autograd.grad(support, tuple(params.values()), create_graph=True)
        virtual = {k: p - group['lr'] * g for (k, p), g in zip(params.items(), grads)}
        if iteration == 0:
            # Commit the classifier update made with the incoming mixer, once.
            committed = {k: v.detach() for k, v in virtual.items()}
            committed_buffers = buffers
            stored = logits[0][:b].detach().clone()
            for target, value in zip(next_state, advanced):
                target[slots] = value.detach()
            support_value, seen_value = support.item(), penalty.item()
        model.eval()
        views = fixed_views if chain_aware else base.mixed_views(xa, unseen, m, w)
        query_logits, _ = base.forward(model, torch.cat([xa, *views]), virtual, query_buffers)
        query = query_logits.chunk(3)
        clean = F.cross_entropy(query[0], ya)
        consistency = base.js(*query)
        objective = clean + consistency
        mixer_optimizer.zero_grad(set_to_none=True)
        hypergrads = torch.autograd.grad(objective, tuple(controller.parameters()))
        grad_norm = torch.stack([g.square().sum() for g in hypergrads]).sum().sqrt()
        if not torch.isfinite(grad_norm) or not torch.isfinite(objective):
            raise RuntimeError('Nonfinite unrolled objective or mixer gradient')
        for p, grad in zip(controller.parameters(), hypergrads):
            p.grad = grad
        updates.append(dict(outer_ce=clean.item(), js=consistency.item(),
                            grad_norm=grad_norm.item()))
        mixer_optimizer.step()
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(committed[name])
        for name, buf in model.named_buffers():
            buf.copy_(committed_buffers[name])
        m, w, _ = score()
    memory.add(raw, labels, stored)
    return next_state, dict(m_after=m.mean().item(), m_after_per_example=m.flatten().tolist(),
        weight_entropy_after=(-(w*w.clamp_min(1e-8).log()).sum(-1)).mean().item(),
        weight_max_after=w.max(-1).values.mean().item(), outer_query_js=updates[-1]['js'],
        classifier_loss=support_value, seen_consistency=seen_value, consistency=kind,
        outer_consistency='js', mixer_updates=updates, presentations=memory.seen,
        solver_variant=variant, fixed_m=None if variant == 'metamix' else fixed_m)


if __name__ == '__main__':
    base.main(unrolled_step=step)
