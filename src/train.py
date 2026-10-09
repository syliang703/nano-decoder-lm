import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def configure_optimizers(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    device_type: str,
) -> torch.optim.AdamW:
    """Configures AdamW optimizer with weight decay applied exclusively to 2D weight matrices.

    1D parameters (biases, LayerNorm scales/biases, embedding parameters) have weight decay set to 0.0.
    """
    decay = []
    no_decay = []

    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    params = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]

    optimizer = torch.optim.AdamW(params=params,
                                  lr=learning_rate,
                                  betas=betas,
                                  )

    return optimizer


def get_lr(
    it: int,
    learning_rate: float,
    min_lr: float,
    warmup_iters: int,
    lr_decay_iters: int,
) -> float:
    """Calculates learning rate at iteration `it` using linear warmup followed by cosine decay."""
    # 1. Linear warmup if it < warmup_iters
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters

    # 2. If it > lr_decay_iters: return min_lr
    if it > lr_decay_iters:
        return min_lr

    # 3. Cosine decay between warmup_iters and lr_decay_iters:
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    grad_clip: float,
    device_type: str,
    ptdtype: torch.dtype,
) -> float:
    """Executes a single forward, backward, gradient clipping, and optimizer step under AMP."""
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type=device_type, dtype=ptdtype):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    return loss.item()