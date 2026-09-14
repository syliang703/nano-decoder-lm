import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticCopyDataset(Dataset):
    """
    Generates synthetic integer sequence pairs where the target is identical to the source.
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
        # TODO: Generate random integer tensors of shape (num_samples, seq_len)
        # Ensure values stay between 4 and vocab_size - 1 so they don't overlap reserved tokens.
        return torch.randint(4, self.vocab_size, (self.num_samples, self.seq_len))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # TODO: Return a single source sequence tensor and target sequence tensor
        return self.data[idx], self.data[idx]

def get_synthetic_dataloader(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    num_samples: int
    ) -> DataLoader:
    # TODO: Instantiate SyntheticCopyDataset
    # TODO: Instantiate and return PyTorch DataLoader with shuffle enabled
    dataset = SyntheticCopyDataset(vocab_size, seq_len, num_samples)
    return DataLoader(dataset, batch_size, shuffle=True)