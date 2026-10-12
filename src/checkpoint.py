from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional, Dict, Any


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    val_loss: float,
    filepath: str | Path,
) -> None:
    """Serializes model state, optimizer state, iteration counter, and validation loss to disk."""
    # TODO: Implement checkpoint dictionary assembly and torch.save
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': iteration,
        "loss": val_loss
    }

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Loads a checkpoint from disk, restoring model and optimizer states.

    Returns the loaded checkpoint dictionary (containing iteration, val_loss, etc.).
    """
    # TODO: Implement torch.load, model.load_state_dict, and optional optimizer state loading
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint
