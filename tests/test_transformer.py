import unittest
import torch
from src.transformer import Transformer

class TestTransformer(unittest.TestCase):
    def test_transformer_forward_shape(self):
        # TODO: Define dimensions (batch_size, src_vocab_size, tgt_vocab_size, d_model, etc.)
        # TODO: Instantiate Transformer model
        # TODO: Create random integer tensors for src and tgt using torch.randint(...)
        # TODO: Run forward pass: output = model(src, tgt)
        # TODO: Assert that output.shape == (batch_size, tgt_seq_len, tgt_vocab_size)
        batch_size = 6
        src_vocab_size = 100
        tgt_vocab_size = 100
        d_model = 64
        num_heads = 4
        num_layers = 2
        d_ff = 128
        max_len = 20
        src_seq_len = 10
        tgt_seq_len = 8
        dropout = 0.1

        model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout)

        src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
        tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

        output = model(src, tgt)

        self.assertEqual(output.shape, (batch_size, tgt_seq_len, tgt_vocab_size))

if __name__ == '__main__':
    unittest.main()