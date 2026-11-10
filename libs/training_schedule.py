"""Epoch cosine cycles reconstructed from checkpoint position."""
import torch


def cosine_for_epoch(optimizer, base_lr, period, completed_epochs):
    """Reconstruct an epoch scheduler from checkpoint position, resetting each cycle."""
    offset = completed_epochs % period
    for group in optimizer.param_groups:
        group['initial_lr'] = base_lr
        if offset == 0:
            group['lr'] = base_lr
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=period, eta_min=0., last_epoch=offset - 1)


