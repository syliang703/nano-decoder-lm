import unittest
import torch
from src.utils import make_src_mask, make_tgt_mask

class TestMaskCreationUtils(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 4
        self.pad_idx = 1
        self.x = torch.tensor([
            [5, 2, 4, 1],  # 1 pad token at the end
            [3, 1, 1, 1]   # 3 pad tokens
        ])

    def test_make_src_mask(self):
        src_mask = make_src_mask(self.x, self.pad_idx)
        self.assertEqual(src_mask.shape, (self.batch_size, 1, 1, self.seq_len))
        self.assertTrue(src_mask[0, 0, 0, 2].item())
        self.assertFalse(src_mask[0, 0, 0, 3].item())

    def test_make_tgt_mask(self):
        tgt_mask = make_tgt_mask(self.x, self.pad_idx)
        self.assertEqual(tgt_mask.shape, (self.batch_size, 1, self.seq_len, self.seq_len))
        self.assertFalse(tgt_mask[0, 0, 0, 1].item())

if __name__ == "__main__":
    unittest.main()