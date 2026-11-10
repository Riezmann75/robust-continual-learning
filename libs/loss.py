"""Experiment objectives; variant-specific masking and scaling are preserved."""
import torch
from torch.nn import functional as F


def barlow_loss(a, b, off_weight=0.0051, eps=1e-05):
    """Batch-standardized cross-correlation; summed BT objective divided by d."""
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] < 2:
        raise ValueError('Need paired Nxd features with N >= 2')
    za = (a - a.mean(0)) / torch.sqrt(a.var(0, unbiased=False) + eps)
    zb = (b - b.mean(0)) / torch.sqrt(b.var(0, unbiased=False) + eps)
    c = za.T @ zb / a.shape[0]
    diagonal = c.diagonal()
    on = (diagonal - 1).square().sum()
    mask = ~torch.eye(c.shape[0], device=c.device, dtype=torch.bool)
    off = c[mask].square().sum()
    loss = (on + off_weight * off) / a.shape[1]
    return (loss, dict(barlow_on=on / a.shape[1], barlow_off=off / a.shape[1], barlow_diag_mean=diagonal.mean(), barlow_off_rms=c[mask].square().mean().sqrt()))


def three_pair_loss(h0, h1, h2, off_weight=0.0051):
    pairs = [barlow_loss(a, b, off_weight) for a, b in ((h0, h1), (h0, h2), (h1, h2))]
    loss = torch.stack([v[0] for v in pairs]).mean()
    metrics = {k: torch.stack([v[1][k] for v in pairs]).mean() for k in pairs[0][1]}
    metrics.update(bt_clean_view1=pairs[0][0], bt_clean_view2=pairs[1][0], bt_view1_view2=pairs[2][0])
    return (loss, metrics)


def js(*logits):
    logps = [F.log_softmax(z, dim=1) for z in logits]
    ps = [p.exp() for p in logps]
    mean = torch.stack(ps).mean(0).clamp_min(1e-08).log()
    return sum(((p * (lp - mean)).sum(1).mean() for p, lp in zip(ps, logps))) / len(ps)


def contrastive_scores(h0, h1, h2, temperature=0.1, labels=None):
    """Three-view scores with same-class nonpositive examples excluded."""
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    n = len(h0)
    if n < 1 or h1.shape != h0.shape or h2.shape != h0.shape:
        raise ValueError('Expected three nonempty aligned feature matrices')
    u = F.normalize(torch.cat((h0, h1, h2)), p=2, dim=1, eps=1e-08)
    scores = u @ u.T / temperature
    ids = torch.arange(n, device=u.device).repeat(3)
    diagonal = torch.eye(3 * n, dtype=torch.bool, device=u.device)
    positives = ids[:, None].eq(ids[None, :]) & ~diagonal
    if labels is None or labels.ndim != 1 or len(labels) != n:
        raise ValueError('One class label per original batch entry is required')
    classes = labels.to(u.device).repeat(3)
    different_class = classes[:, None].ne(classes[None, :])
    allowed = positives | different_class
    return (scores.masked_fill(~allowed, -torch.inf), positives)


def infonce_loss(h0, h1, h2, temperature=0.1, labels=None):
    """Average log probability of both positives for every class-aware anchor."""
    scores, positives = contrastive_scores(h0, h1, h2, temperature, labels)
    normalizer = torch.logsumexp(scores, dim=1)
    positive_score = scores[positives].view(len(scores), 2).mean(1)
    return (normalizer - positive_score).mean()


def consistency_loss(logits, features, labels, kind, temperature):
    if kind == 'js':
        return js(*logits)
    return infonce_loss(*features, temperature=temperature, labels=labels)


def feature_loss(h0, h1, h2):
    u0, u1, u2 = [F.normalize(h, p=2, dim=1, eps=1e-08) for h in (h0, h1, h2)]
    return 0.5 * ((u0 - u1).square().sum(1).mean() + (u0 - u2).square().sum(1).mean())


def instance_contrastive_scores(h0, h1, h2, temperature=0.1):
    """Three-view scores with all other instances used as negatives."""
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    n = len(h0)
    if n < 1 or h1.shape != h0.shape or h2.shape != h0.shape:
        raise ValueError('Expected three nonempty aligned feature matrices')
    u = F.normalize(torch.cat((h0, h1, h2)), p=2, dim=1, eps=1e-08)
    scores = u @ u.T / temperature
    ids = torch.arange(n, device=u.device).repeat(3)
    diagonal = torch.eye(3 * n, dtype=torch.bool, device=u.device)
    positives = ids[:, None].eq(ids[None, :]) & ~diagonal
    return (scores.masked_fill(diagonal, -torch.inf), positives)


def instance_infonce_loss(h0, h1, h2, temperature=0.1):
    scores, positives = instance_contrastive_scores(h0, h1, h2, temperature)
    normalizer = torch.logsumexp(scores, dim=1)
    positive_score = scores[positives].view(len(scores), 2).mean(1)
    return (normalizer - positive_score).mean()


def feature_mse_loss(h0, h1, h2):
    u0, u1, u2 = [F.normalize(h, p=2, dim=1, eps=1e-08) for h in (h0, h1, h2)]
    return 0.5 * ((u0 - u1).square().sum(1).mean() + (u0 - u2).square().sum(1).mean())


def head_ce(z, y):
    return F.cross_entropy(z.reshape(-1, 10), y[:, None].expand(-1, z.shape[1]).reshape(-1))


def ensemble(z):
    return z.softmax(-1).mean(1)


def diversity_loss(h):
    u = F.normalize(h, dim=-1, eps=1e-08)
    gram = u @ u.transpose(1, 2)
    mask = ~torch.eye(h.shape[1], device=h.device, dtype=torch.bool)
    return gram[:, mask].square().mean()


def seen_unseen_prediction_js(mixed, targets):
    """Mean pairwise JS over views, target chains and examples (class axis last)."""
    if (mixed.ndim != 3 or targets.ndim != 4
            or targets.shape[0] != mixed.shape[0] or targets.shape[2:] != mixed.shape[1:]):
        raise ValueError('Expected mixed logits (V,N,C) and target logits (V,K,N,C)')
    classes = mixed.shape[-1]
    paired = mixed[:, None].expand_as(targets)
    return js(paired.reshape(-1, classes), targets.detach().reshape(-1, classes))


def seen_unseen_feature_loss(mixed_features, target_features):
    """Raw-feature MSE: (V,N,D) seen mixtures vs (V,K,N,D) unseen chains.

    Every seen view matches all K independently sampled targets for that view.
    Mean reduction covers views, targets, examples and feature coordinates.
    """
    if (mixed_features.ndim != 3 or target_features.ndim != 4
            or target_features.shape[0] != mixed_features.shape[0]
            or target_features.shape[2:] != mixed_features.shape[1:]):
        raise ValueError('Expected mixed (V,N,D) and targets (V,K,N,D)')
    return (mixed_features[:, None] - target_features.detach()).square().mean()
