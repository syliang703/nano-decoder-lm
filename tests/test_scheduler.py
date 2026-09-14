import unittest
import torch
from src.scheduler import TransformerLRScheduler

class TestTransformerLRScheduler(unittest.TestCase):
    def setUp(self):
        # Simple parameter for dummy optimizer
        self.model = torch.nn.Linear(10, 10)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0)
        self.d_model = 512
        self.warmup_steps = 4000

    def test_lr_increases_during_warmup(self):
        scheduler = TransformerLRScheduler(self.optimizer, self.d_model, self.warmup_steps)

        # TODO 1: Step the scheduler once and capture lr1 (scheduler.get_last_lr()[0])
        # TODO 2: Step the scheduler again and capture lr2
        # TODO 3: Assert lr2 > lr1 (learning rate should increase during warmup)
        scheduler.step()
        lr1 = scheduler.get_last_lr()[0]
        scheduler.step()
        lr2 = scheduler.get_last_lr()[0]
        self.assertGreater(lr2, lr1)

    def test_lr_decreases_after_warmup(self):
        scheduler = TransformerLRScheduler(self.optimizer, self.d_model, self.warmup_steps)

        # TODO 1: Fast-forward step count past warmup (e.g., set scheduler._step_count = 5000)
        # TODO 2: Step the scheduler once and capture lr_before
        # TODO 3: Step the scheduler again and capture lr_after
        # TODO 4: Assert lr_after < lr_before (learning rate should decrease after warmup)
        scheduler._step_count = 5000
        scheduler.step()
        lr_before = scheduler.get_last_lr()[0]
        scheduler.step()
        lr_after = scheduler.get_last_lr()[0]
        self.assertLess(lr_after, lr_before)

if __name__ == "__main__":
    unittest.main()