import numpy as np

import torch


class CosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing LR schedule (used in Carmon et al, 2019).
    """
    def __init__(self, optimizer, max_lr, epochs, last_epoch=-1):
        self.max_lr = max_lr
        self.epochs = epochs
        self._reset()
        super(CosineLR, self).__init__(optimizer, last_epoch)
    
    def _reset(self):
        self.current_lr = self.max_lr
        self.current_epoch = 1

    def step(self):
        self.current_lr = self.max_lr * 0.5 * (1 + np.cos((self.current_epoch - 1) / self.epochs * np.pi))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
        self.current_epoch += 1
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        
    def get_lr(self):
        return self.current_lr


class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * (self.last_epoch + 1) / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
