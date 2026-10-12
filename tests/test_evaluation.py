import numpy as np
import os
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import PretrainedDataset
from src.model import NanoTransformer
from src.train import estimate_loss

@pytest.fixture
def dummy_bin_file(tmp_path) -> str:
    """Fixture creating a temporary dummy token binary file for testing."""
    bin_path = os.path.join(tmp_path, "test_tokens.bin")
    # Create a simple deterministic sequence of integers: [0, 1, 2, ..., 99]
    tokens = np.arange(100, dtype=np.uint16)
    tokens.tofile(bin_path)
    return bin_path

def test_estimate_loss_preserves_weights(dummy_bin_file):
    """Verifies that estimate_loss computes evaluation loss without modifying model weights."""
    # TODO: Initialize model and dummy eval data loader
    vocab_size = 100

    model = NanoTransformer(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        max_len=128,
        dropout=0.0,
    )

    block_size = 10
    eval_data = PretrainedDataset(bin_path=dummy_bin_file, block_size=block_size)
    eval_dataloader = DataLoader(eval_data, batch_size=2)

    # TODO: Capture original model weights state dict
    original_model_weights = {k: v.clone() for k, v in model.state_dict().items()}

    # TODO: Run estimate_loss
    avg_loss = estimate_loss(model=model,
                             eval_dataloader=eval_dataloader,
                             device_type='cpu',
                             ptdtype=torch.float32,
                             max_eval_iters=5
                             )
    # TODO: Assert model weights are identical and output loss is a valid float
    current_weights = model.state_dict()

    for key in original_model_weights:
        assert torch.equal(original_model_weights[key], current_weights[key])

    assert isinstance(avg_loss, float)