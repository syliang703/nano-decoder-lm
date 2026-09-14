import torch
import torch.nn as nn

def get_label_smoothed_ce_loss(pad_idx: int, label_smoothing: float = 0.1) -> nn.CrossEntropyLoss:
    """
    Constructs a PyTorch CrossEntropyLoss instance configured for sequence-to-sequence training.

    Args:
        pad_idx: The integer index representing the padding token (must be ignored in loss calculation).
        label_smoothing: The float value for label smoothing regularization.

    Returns:
        An instance of nn.CrossEntropyLoss with padding ignored and label smoothing enabled.
    """
    # TODO: Instantiate and return nn.CrossEntropyLoss with the appropriate parameters
    return nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=label_smoothing)