import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PretrainedDataset(Dataset):
    """Memory-mapped PyTorch Dataset for high-throughput language model pre-training.

    Lazily reads contiguous token sequences from binary NumPy arrays on disk
    without loading the entire dataset into RAM.
    """

    def __init__(
        self,
        bin_path: str,
        block_size: int,
        dtype: np.dtype = np.uint16,
    ):
        """Initializes the memory-mapped token array and computes valid sample index range.

        Args:
            bin_path: Path to the serialized binary file (.bin).
            block_size: Sequence length (S) for input context window.
            dtype: NumPy integer data type matching binary file serialization (default uint16).
        """
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary dataset file not found at: {bin_path}")

        self.bin_path = bin_path
        self.block_size = block_size
        self.dtype = dtype
        self.data = np.memmap(bin_path, dtype=dtype, mode='r') # Read-only mode ('r')
        self.total_tokens = len(self.data)

        if self.total_tokens <= self.block_size:
            raise ValueError(f"Total number of tokens too small for single sequence window")


    def __len__(self) -> int:
        """Returns the total number of valid starting indices in the dataset."""
        return self.total_tokens - self.block_size


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetches a single sample window and its target tokens shifted by 1 position.

        Args:
            idx: Index of starting token position.

        Returns:
            Tuple of (x, y) where:
                x: Input tensor of shape (block_size,) cast to torch.int64 (long)
                y: Target tensor of shape (block_size,) cast to torch.int64 (long)
        """
        # Boundary safety / out-of-bounds error checking
        if idx < 0:
            idx = idx + len(self)

        if idx < 0 or idx >= self.__len__():
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self)}")

        block = self.data[idx : idx + self.block_size + 1]

        input_np = block[:-1]
        target_np = block[1:]

        input_tensor = torch.from_numpy(input_np.astype(np.int64))
        target_tensor = torch.from_numpy(target_np.astype(np.int64))

        return input_tensor, target_tensor