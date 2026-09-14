import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticCopyDataset(Dataset):
    """
    Generates synthetic integer sequences for autoregressive language model training.
    """
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int, pad_idx: int = 1, bos_idx: int = 2, eos_idx: int = 3):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        # Reserved special tokens (0=unused/pad, 1=pad, 2=bos, 3=eos)
        self.reserved_indices = {0, pad_idx, bos_idx, eos_idx}
        self.data = self._generate_data()

    def _generate_data(self):
        # Generate random integer tensors of shape (num_samples, seq_len + 1)
        # Values stay between 4 and vocab_size - 1 to avoid special token overlap
        return torch.randint(4, self.vocab_size, (self.num_samples, self.seq_len + 1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return a single sequence tensor for autoregressive slicing in train_one_epoch
        return self.data[idx]

def get_synthetic_dataloader(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    num_samples: int
    ) -> DataLoader:
    """
    Instantiates SyntheticCopyDataset and returns a DataLoader
    """
    dataset = SyntheticCopyDataset(vocab_size, seq_len, num_samples)
    return DataLoader(dataset, batch_size, shuffle=True)