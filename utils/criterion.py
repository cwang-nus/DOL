import torch
import torch.nn as nn
import numpy as np

def select_criterion(loss_type):
    if loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'maskedmae':
        criterion = MaskedMAE()
    elif loss_type == 'smoothl1':
        criterion = nn.SmoothL1Loss()
    return criterion

def masked_mae(preds, labels, null_val=np.nan, reduce = True):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if reduce:
        return torch.mean(loss)
    else:
        return loss
class MaskedMAE(nn.Module):
    def __init__(self):
        super(MaskedMAE, self).__init__()

    def forward(self, preds, labels, null_val=np.nan):
        assert preds.shape == labels.shape
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels!=null_val)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(preds-labels)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

class CosineWarmupScheduler():
    def __init__(self, optimizer, d_model, n_warmup_steps, lr_mul=1.0):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_periodic_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        self._update_lr()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        if n_steps <= self.n_warmup_steps:
            return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))
        else:
            base = (d_model ** -0.5) * n_warmup_steps ** (-0.5) * (1 + np.cos(
                np.pi * ((n_steps - self.n_warmup_steps) % self.n_periodic_steps) / self.n_periodic_steps))
            return base

    def _update_lr(self):
        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr