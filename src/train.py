import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import make_src_mask, make_tgt_mask

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    loss_fn: nn.Module,
    pad_idx: int,
    device: torch.device
) -> float:
    """
    Runs one training epoch over the dataloader.

    Returns:
        Average loss over all batches in the epoch.
    """
    model.train()
    total_loss = 0.0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        # Slice target tensor for target input and target labels (teacher forcing)
        # TODO 1: Extract tgt_input (all columns except the last)
        # TODO 2: Extract tgt_label (all columns except the first)
        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]

        # Generate masks
        # TODO 3: Create src_mask using make_src_mask(src, pad_idx)
        # TODO 4: Create tgt_mask using make_tgt_mask(tgt_input, pad_idx)
        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx)

        # Forward pass
        # TODO 5: Pass src, tgt_input, src_mask, tgt_mask into model(...)
        output = model(src, tgt_input, src_mask, tgt_mask)

        # Calculate loss
        # TODO 6: Flatten logits to (N, vocab_size) and tgt_label to (N,)
        # TODO 7: Compute scalar loss using loss_fn
        vocab_size = output.size(-1)
        output = output.view(-1, vocab_size)
        tgt_label = tgt_label.reshape(-1)
        loss = loss_fn(output, tgt_label)

        # Backward pass & optimization
        # TODO 8: Zero gradients, execute backward pass, step optimizer, step scheduler
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)