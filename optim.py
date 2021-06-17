import numpy as np
from torch import nn


class TransformerSchedule(nn.Module):
    """A simple wrapper class for learning rate scheduling
    """
    def __init__(self, optimizer, init_lr, d_model, warmup_steps):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.cur_steps = 0
        self.cur_lr = init_lr

    def step(self):
        self.update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr(self):
        return self.cur_lr

    def get_lr_scale(self):
        d_model = self.d_model
        cur_steps, warmup_steps = self.cur_steps, self.warmup_steps
        return (d_model ** -0.5) * min(cur_steps ** (-0.5), cur_steps * warmup_steps ** (-1.5))

    def update_learning_rate(self):
        r"""Learning rate scheduling per step"""
        self.cur_steps += 1
        lr = self.init_lr * self.get_lr_scale()
        self.cur_lr = lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class CosineSchedule():
    r"""A simple wrapper class for learning rate scheduling
    """

    def __init__(self, optimizer, init_lr, n_warmup_steps, n_training_steps):
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.n_warmup_steps = n_warmup_steps
        self.n_training_steps = n_training_steps
        self.n_steps = 0
        self.cur_lr = init_lr

    def zero_grad(self):
        self._optimizer.zero_grad()

    def state_dict(self):
        return self._optimizer.state_dict(), self.n_steps

    def load_state_dict(self, state_dict):
        opt, self.n_steps = state_dict
        self._optimizer.load_state_dict(opt)

    def get_lr(self):
        return self.cur_lr

    def step(self):
        self._update_learning_rate()
        self._optimizer.step()

    def _get_lr_scale(self):
        if self.n_steps < self.n_warmup_steps:
            return float(self.n_steps) / float(max(1, self.n_warmup_steps))
        progress = float(self.n_steps - self.n_warmup_steps) / float(
            max(1, self.n_training_steps - self.n_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def _update_learning_rate(self):
        r"""Learning rate scheduling per step"""
        self.n_steps += 1
        lr = self._get_lr_scale() * self.init_lr
        self.cur_lr = lr

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
