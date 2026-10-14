from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from src.model import NanoTransformer


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    val_loss: float,
    filepath: str | Path,
    model_args: Optional[dict] = None
) -> None:
    """Serializes model state, optimizer state, iteration counter, and validation loss to disk."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': iteration,
        'loss': val_loss
    }

    if model_args is not None:
        checkpoint["model_args"] = model_args

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str | Path,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> tuple[nn.Module, Dict[str, Any]]:
    """Loads a checkpoint from disk, restoring model and optimizer states.

    Returns the loaded checkpoint dictionary (containing iteration, val_loss, etc.).
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    if model is None:
        if "model_args" in checkpoint:
            model = NanoTransformer(**checkpoint["model_args"]).to(device)
        else:
            raise ValueError("Model not provided and model arguments not found in checkpoint data. ")

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, checkpoint