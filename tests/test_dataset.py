import torch
import pytest
from src.dataset import get_synthetic_dataloader

@pytest.fixture
def dataloader_params():
    return {
        "vocab_size": 50,
        "seq_len": 10,
        "batch_size": 4,
        "num_samples": 16,
    }

def test_dataloader_batch_shape(dataloader_params):
    dataloader = get_synthetic_dataloader(
        vocab_size=dataloader_params["vocab_size"],
        seq_len=dataloader_params["seq_len"],
        batch_size=dataloader_params["batch_size"],
        num_samples=dataloader_params["num_samples"],
    )

    # 1. Get the first batch from the dataloader
    batch = next(iter(dataloader))

    # 2. Assert batch shape is (batch_size, seq_len + 1)
    expected_shape = (dataloader_params["batch_size"], dataloader_params["seq_len"] + 1)
    assert batch.shape == expected_shape

    # 3. Assert tensor data type is integer (torch.long or torch.int64)
    assert batch.dtype == torch.long