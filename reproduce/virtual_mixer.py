"""One-step bilevel SGD for chain selection on Split-CIFAR10."""
import numpy as np
import torch
from torch.nn import functional as F

import chain_aware_mixing as base


def split_task_indices(labels, tasks):
    """Reserve 10% per class for mixer queries, disjoint from classifier/replay."""
    rng = np.random.default_rng(17)
    support, queries = [], []
    for pair in tasks:
        train, query = [], []
        for label in pair:
            ids = rng.permutation(np.flatnonzero(labels == label))
            n = max(1, len(ids) // 10)
            query.extend(ids[:n].tolist())
            train.extend(ids[n:].tolist())
        support.append(train)
        queries.append(query)
    return support, queries


def meta_loss(model, inner_loss, lr, query, unseen_query, labels):
    """Differentiate a plain SGD step, then score independent query images."""
    params = dict(model.named_parameters())
    grads = torch.autograd.grad(inner_loss, tuple(params.values()), create_graph=True)
    virtual = {key: p - lr * grad for (key, p), grad in zip(params.items(), grads)}
    model.eval()
    logits, _ = base.forward(model, torch.cat((query, unseen_query)),
                             virtual, base.clone_buffers(model))
    clean, unseen = logits.chunk(2)
    return F.cross_entropy(clean, labels), F.cross_entropy(unseen, labels), grads


def step(model, controller, optimizer, mixer_optimizer, memory, state, raw,
         labels, device, *, query_batch, kind, temperature, current_capacity=32, **options):
    # The runner constructs both optimizers as plain SGD with one parameter group.
    x = base.tensor(raw, device)
    y = torch.as_tensor(labels, device=device)
    b = len(x)
    replay = memory.sample(32, device, raw=True) if memory.size else None
    distill = memory.sample(32, device) if memory.size else None
    context_replay = (base.tensor(replay[0], device), *replay[1:]) if replay else None
    context = base.context_for(model, x, y, context_replay)
    raw_all = np.concatenate((raw, replay[0])) if replay else raw
    xa = base.tensor(raw_all, device)
    ya = torch.cat((y, replay[1])) if replay else y
    slots = base.active_slots(b, len(xa) - b, device, current_capacity)
    incoming = tuple(v[slots].detach() for v in state)
    perm = np.random.permutation(9)
    cut = np.random.randint(5, 8)
    seen = base.chains(raw_all, [base.ops.augmentations[i] for i in perm[:cut]], device)
    query_raw, query_labels = query_batch
    unseen = base.chains(query_raw, [base.ops.augmentations[i] for i in perm[cut:]],
                         device, views_count=1, branch_count=1)[0][0]
    candidates = base.candidate_descriptors(model, xa, seen)
    m, w, advanced = controller(context, incoming, candidates)
    model.train()
    z, h = base.forward(model, torch.cat((xa, *base.mixed_views(xa, seen, m, w))))
    logits = z.chunk(3)[0]
    seen_loss = base.consistency_loss(z.chunk(3), h.chunk(3), ya, kind, temperature)
    inner = F.cross_entropy(logits[:b], y) + seen_loss
    if replay:
        inner = inner + .5 * F.cross_entropy(logits[b:], replay[1])
        inner = inner + .2 * F.mse_loss(base.forward(model, distill[0])[0], distill[2])

    clean_ce, unseen_ce, inner_grads = meta_loss(
        model, inner, optimizer.param_groups[0]['lr'], base.tensor(query_raw, device),
        unseen, torch.as_tensor(query_labels, device=device))
    outer = clean_ce + unseen_ce
    mixer_grads = torch.autograd.grad(outer, tuple(controller.parameters()))
    if not all(torch.isfinite(t.detach()).all() for t in (inner, outer, *inner_grads, *mixer_grads)):
        raise RuntimeError('Nonfinite bilevel loss or gradient')

    # Commit precisely the support-only SGD step used to construct virtual parameters.
    for p, grad in zip(model.parameters(), inner_grads):
        p.grad = grad.detach()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for p, grad in zip(controller.parameters(), mixer_grads):
        p.grad = grad.detach()
    mixer_optimizer.step()
    mixer_optimizer.zero_grad(set_to_none=True)
    memory.add(raw, labels, logits[:b].detach())
    next_state = tuple(v.detach().clone() for v in state)
    with torch.no_grad():
        for dest, value in zip(next_state, advanced):
            dest[slots] = value
        after_m, after_w, _ = controller(context, incoming, candidates)
    return next_state, dict(
        consistency=kind, outer_consistency='virtual_query_ce', seen_consistency=seen_loss.item(),
        classifier_loss=inner.item(), query_clean_ce=clean_ce.item(), query_unseen_ce=unseen_ce.item(),
        mixer_loss=outer.item(), meta_grad_norm=torch.stack([g.norm() for g in mixer_grads]).norm().item(),
        m_after=after_m.mean().item(), m_after_per_example=after_m.flatten().tolist(),
        weight_entropy_after=(-(after_w * after_w.clamp_min(1e-8).log()).sum(-1)).mean().item(),
        weight_max_after=after_w.max(-1).values.mean().item(), presentations=memory.seen)


if __name__ == '__main__':
    base.main(bilevel_step=step, split_indices=split_task_indices)
