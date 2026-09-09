import unittest
import torch
from src.modules import PositionwiseFeedForward, LayerNorm

class TestTransformerModules(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 4
        self.d_model = 8
        self.d_ff = 32
        self.x = torch.randn(self.batch_size, self.seq_len, self.d_model)

    def test_feedforward_shape(self):
        ffn = PositionwiseFeedForward(d_model=self.d_model, d_ff=self.d_ff)
        out = ffn(self.x)
        # TODO 1: Assert that the output shape matches the input shape
        # Pass both shapes into self.assertEqual()
        self.assertEqual(self.x.shape, out.shape)

    def test_layernorm_properties(self):
        ln = LayerNorm(features=self.d_model)
        out = ln(self.x)

        # Verify output shape retention
        self.assertEqual(out.shape, self.x.shape)

        # Calculate mean and variance across the feature dimension (dim=-1)
        mean = out.mean(dim=-1)
        var = out.var(dim=-1, unbiased=False)

        # TODO 2: Assert that calculated mean is close to 0 and variance is close to 1
        # Hint: Use torch.allclose with atol=1e-3 against tensors of zeros/ones
        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-3))
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-3))

if __name__ == "__main__":
    unittest.main()