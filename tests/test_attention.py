import unittest
import torch
from src.attention import scaled_dot_product_attention

class TestScaledDotProductAttention(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.batch_size = 2
        self.num_heads = 4
        self.seq_len = 6
        self.d_k = 16
        self.d_v = 16

        torch.manual_seed(42)
        # TODO: Initialize random tensors for Q, K, V on self.device
        self.Q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.d_k, device=self.device)
        self.K = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.d_k, device=self.device)
        self.V = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.d_v, device=self.device)

    def test_output_and_weights_shape(self):
        """Test output and weight tensor dimensions match expectation."""
        # TODO 1: Call scaled_dot_product_attention with self.Q, self.K, self.V
        output, weights = scaled_dot_product_attention(self.Q, self.K, self.V)

        # TODO 2: Assert output shape == (batch_size, num_heads, seq_len, d_v)
        self.assertEqual(output.shape, (self.batch_size, self.num_heads, self.seq_len, self.d_v))
        # TODO 3: Assert weights shape == (batch_size, num_heads, seq_len, seq_len)
        self.assertEqual(weights.shape, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))

    def test_softmax_row_sums_to_one(self):
        """Test that attention weights sum to 1.0 along the last dimension."""
        # TODO 1: Run attention and compute row sums along dim=-1
        _, weights = scaled_dot_product_attention(self.Q, self.K, self.V)
        sums = torch.sum(weights, dim=-1)
        # TODO 2: Use torch.allclose() to assert all sums equal 1.0
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums)))

    def test_causal_masking(self):
        """Test that causal masking zeroes out attention weights for future tokens."""
        # TODO 1: Create a boolean lower-triangular mask of shape (seq_len, seq_len)
        # Hint: torch.tril(...)
        causal_mask = torch.tril(torch.ones(self.seq_len, self.seq_len, dtype=torch. bool, device=self.device))

        # TODO 2: Pass mask into scaled_dot_product_attention
        _, weights = scaled_dot_product_attention(self.Q, self.K, self.V, causal_mask)

        # TODO 3: Assert that values in the upper triangle (future tokens) are 0.0
        self.assertTrue(torch.allclose(torch.triu(weights, diagonal=1), torch.zeros_like(weights)))

if __name__ == "__main__":
    unittest.main()