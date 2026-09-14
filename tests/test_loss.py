import unittest
import torch
from src.loss import get_label_smoothed_ce_loss

class TestLabelSmoothedLoss(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 3
        self.vocab_size = 10
        self.pad_idx = 0
        self.label_smoothing = 0.1

        # Logits of shape (batch_size, seq_len, vocab_size)
        self.logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size)

        # Targets containing pad_idx (0)
        self.targets = torch.tensor([
            [5, 2, 0],
            [3, 0, 0]
        ])

    def test_loss_shape_and_execution(self):
        loss_fn = get_label_smoothed_ce_loss(self.pad_idx, self.label_smoothing)

        # TODO 1: Reshape self.logits to 2D (batch_size * seq_len, vocab_size) and self.targets to 1D (batch_size * seq_len) using .view()
        # TODO 2: Compute the loss value using loss_fn
        # TODO 3: Assert that the loss shape is a 0D scalar (i.e., loss.dim() == 0)
        # TODO 4: Assert that loss.item() is greater than 0.0
        self.logits = self.logits.view(-1, self.vocab_size)
        self.targets = self.targets.view(-1)
        loss = loss_fn(self.logits, self.targets)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)

if __name__ == "__main__":
    unittest.main()