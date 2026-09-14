import torch

def make_causal_mask(x: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Creates a combined causal (subsequent) and padding mask for decoder-only input sequences.

    Args:
        x: Input tensor of shape (batch_size, seq_len)
        pad_idx: Integer index representing padding token to ignore

    Returns:
        Tensor of shape (batch_size, 1, seq_len, seq_len) where True (1) indicates valid/allowed attention positions and False (0) indicates masked positions.
    """
    seq_len = x.size(1)

    # 1. Padding mask: (batch_size, 1, 1, seq_len)
    pad_mask = (x != pad_idx).unsqueeze(1).unsqueeze(2)

    # 2. Lower-triangular Causal mask: (1, 1, seq_len, seq_len)
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool().unsqueeze(0).unsqueeze(0)

    # 3. Combine both masks via logical AND
    return pad_mask & causal_mask