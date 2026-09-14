import pytest
import torch
from src.transformer import NanoTransformer
from src.utils import make_causal_mask

def test_nano_transformer_forward_shape():
    """Verify that input token indices map to (Batch, Seq_Len, Vocab_Size) logits."""
    batch_size = 2
    seq_len = 8
    vocab_size = 100
    d_model = 64

    # Use correct parameter names (num_heads, num_layers) matching NanoTransformer
    model = NanoTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=2,
        num_layers=2
    )
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(idx)
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Expected shape {(batch_size, seq_len, vocab_size)}, got {logits.shape}"

def test_causal_mask_prevents_future_attention():
    """Ensure causal mask correctly masks out future tokens."""
    seq_len = 4
    dummy_input = torch.zeros((1, seq_len), dtype=torch.long)
    mask = make_causal_mask(dummy_input, pad_idx=1)

    # Upper triangle (excluding diagonal) should be False indicating masked positions
    assert not mask[0, 0, 0, 1]
    # Diagonal and lower triangle should be True indicating unmasked positions
    assert mask[0, 0, 0, 0]