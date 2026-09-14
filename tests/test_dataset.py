import unittest
import torch
from src.dataset import get_synthetic_dataloader

class TestSyntheticDataset(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 50
        self.seq_len = 10
        self.batch_size = 4
        self.num_samples = 16

    def test_dataloader_batch_shapes(self):
        dataloader = get_synthetic_dataloader(
            self.vocab_size, self.seq_len, self.batch_size, self.num_samples
        )

        # 1. Get the first batch from the dataloader
        # 2. Assert src and tgt shapes are (batch_size, seq_len)
        # 3. Assert tensor data types are integer (torch.long or torch.int64)
        src, tgt = next(iter(dataloader))
        self.assertEqual(src.shape, (self.batch_size, self.seq_len))
        self.assertEqual(tgt.shape, (self.batch_size, self.seq_len))
        self.assertEqual(src.dtype, torch.long)
        self.assertEqual(tgt.dtype, torch.long)

if __name__ == "__main__":
    unittest.main()