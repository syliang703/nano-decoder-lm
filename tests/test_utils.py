import torch
import pytest
from src.utils import make_causal_mask

@pytest.fixture
def mask_inputs():
    batch_size = 2
    seq_len = 4
    pad_idx = 1
    x = torch.tensor([
        [5, 2, 4, 1],  # 1 pad token at the end
        [3, 1, 1, 1]   # 3 pad tokens
    ])
    return x, batch_size, seq_len, pad_idx

def test_make_causal_mask(mask_inputs):
    x, batch_size, seq_len, pad_idx = mask_inputs
    causal_mask = make_causal_mask(x, pad_idx)

    # 1. Assert shape is (batch_size, 1, seq_len, seq_len)
    assert causal_mask.shape == (batch_size, 1, seq_len, seq_len)

    # 2. Check padding masking (last column of first sample should be False due to pad_idx=1)
    assert not causal_mask[0, 0, 0, 3].item()

    # 3. Check causal lower-triangular constraint (above diagonal should be False)
    # Row 0, Col 1 should be False (can't look into the future)
    assert not causal_mask[0, 0, 0, 1].item()

    # Row 1, Col 0 should be True (valid unmasked position for allowed token)
    assert causal_mask[0, 0, 1, 0].item()