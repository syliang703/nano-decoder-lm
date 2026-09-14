import torch

def make_src_mask(src: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Creates a padding mask for the source sequence.

    Args:
        src: Tensor of shape (batch_size, src_seq_len)
        pad_idx: The integer index representing the padding token.

    Returns:
        Tensor of shape (batch_size, 1, 1, src_seq_len) where
        True (or 1) indicates valid tokens and False (or 0) indicates pad tokens.
    """
    # TODO: Generate boolean mask checking src != pad_idx and unsqueeze dimensions for broadcasting
    src_mask = src != pad_idx
    return src_mask.unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Creates a combined causal (subsequent) and padding mask for the target sequence.

    Args:
        tgt: Tensor of shape (batch_size, tgt_seq_len)
        pad_idx: The integer index representing the padding token.

    Returns:
        Tensor of shape (batch_size, 1, tgt_seq_len, tgt_seq_len)
    """
    # TODO 1: Generate padding mask for tgt of shape (batch_size, 1, 1, tgt_seq_len)
    # TODO 2: Generate causal mask of shape (1, 1, tgt_seq_len, tgt_seq_len) using torch.tril
    # TODO 3: Combine both masks using element-wise logical AND (&)
    pad_mask = make_src_mask(tgt, pad_idx)
    tgt_seq_len = tgt.size(1)
    causal_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len, device=tgt.device)).bool()
    return pad_mask & causal_mask