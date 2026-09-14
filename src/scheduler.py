import torch
from torch.optim.lr_scheduler import _LRScheduler

class TransformerLRScheduler(_LRScheduler):
    """
    Learning rate scheduler matching the Attention Is All You Need paper formula.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int = 4000, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # PyTorch tracks step count via self._step_count
        step = max(1, self._step_count)

        # TODO: Calculate learning rate multiplier based on the formula:
        # lr = (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
        # Return a list of learning rates (one for each parameter group in the optimizer)
        # Hint: [base_lr * lr_scale for base_lr in self.base_lrs]
        lr_scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return [lr * lr_scale for lr in self.base_lrs]