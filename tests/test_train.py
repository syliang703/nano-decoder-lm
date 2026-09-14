import unittest
import torch
import torch.nn as nn
from src.dataset import get_synthetic_dataloader
from src.loss import get_label_smoothed_ce_loss
from src.scheduler import TransformerLRScheduler
from src.train import train_one_epoch

# Minimal dummy model for testing the training loop step
class DummyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt_input, src_mask, tgt_mask):
        # Maps input tokens to logits of shape (batch_size, seq_len, vocab_size)
        x = self.embedding(tgt_input)
        return self.fc(x)

class TestTrainEpoch(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 20
        self.seq_len = 8
        self.batch_size = 2
        self.num_samples = 4
        self.d_model = 16
        self.pad_idx = 1
        self.device = torch.device("cpu")

    def test_train_one_epoch_execution(self):
        dataloader = get_synthetic_dataloader(
            self.vocab_size, self.seq_len, self.batch_size, self.num_samples
        )
        model = DummyTransformer(self.vocab_size, self.d_model).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        scheduler = TransformerLRScheduler(optimizer, self.d_model, warmup_steps=10)
        loss_fn = get_label_smoothed_ce_loss(self.pad_idx)

        # TODO 1: Run train_one_epoch(...) and capture the returned average loss
        # TODO 2: Assert that loss is an instance of float
        # TODO 3: Assert that loss is > 0.0
        loss = train_one_epoch(model, dataloader, optimizer, scheduler, loss_fn, self.pad_idx, self.device)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)

if __name__ == "__main__":
    unittest.main()