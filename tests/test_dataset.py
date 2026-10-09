import os
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from src.dataset import PretrainedDataset


@pytest.fixture
def dummy_bin_file(tmp_path) -> str:
    """Fixture creating a temporary dummy token binary file for testing."""
    bin_path = os.path.join(tmp_path, "test_tokens.bin")
    # Create a simple deterministic sequence of integers: [0, 1, 2, ..., 99]
    tokens = np.arange(100, dtype=np.uint16)
    tokens.tofile(bin_path)
    return bin_path


def test_dataset_initialization_and_length(dummy_bin_file):
    """Tests proper dataset length computation based on block_size."""
    block_size = 10
    dataset = PretrainedDataset(bin_path=dummy_bin_file, block_size=block_size)

    assert dataset.total_tokens == 100
    assert len(dataset) == dataset.total_tokens - block_size


def test_dataset_autoregressive_target_shift(dummy_bin_file):
    """Tests that target sequence y is precisely offset by 1 position relative to x."""
    block_size = 5
    dataset = PretrainedDataset(bin_path=dummy_bin_file, block_size=block_size)

    input_sample, target_sample = dataset[0]

    # TODO: Assert input and target match expected tensor
    assert torch.equal(input_sample, torch.tensor([0, 1, 2, 3, 4]))
    assert torch.equal(target_sample, torch.tensor([1, 2, 3, 4, 5]))

    # TODO: Assert input and target shape and dtype
    assert input_sample.shape == (block_size, )
    assert input_sample.dtype == torch.int64

    assert target_sample.shape == (block_size, )
    assert target_sample.dtype == torch.int64


def test_dataset_out_of_bounds_safety(dummy_bin_file):
    """Tests that requesting an invalid index raises an IndexError."""
    block_size = 10
    dataset = PretrainedDataset(bin_path=dummy_bin_file, block_size=block_size)

    with pytest.raises(IndexError):
        _ = dataset[-999]

    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]

    with pytest.raises(IndexError):
        _ = dataset[999]


def test_dataloader_batch_collation(dummy_bin_file):
    """Tests integration with PyTorch DataLoader for batch shape collation."""
    block_size = 8
    batch_size = 4
    dataset = PretrainedDataset(bin_path=dummy_bin_file, block_size=block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    input_batch, target_batch = next(iter(dataloader))

    # Check input and target batch shapes
    assert input_batch.shape == (batch_size, block_size)
    assert target_batch.shape == (batch_size, block_size)