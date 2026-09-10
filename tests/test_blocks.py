import unittest
import torch
from src.blocks import EncoderBlock, DecoderBlock

class TestTransformerBlocks(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.src_len = 5
        self.tgt_len = 4
        self.d_model = 8
        self.heads = 2
        self.d_ff = 16

        self.x_src = torch.randn(self.batch_size, self.src_len, self.d_model)
        self.x_tgt = torch.randn(self.batch_size, self.tgt_len, self.d_model)

    def test_encoder_block_shape(self):
        encoder_block = EncoderBlock(d_model=self.d_model, heads=self.heads, d_ff=self.d_ff)
        out = encoder_block(self.x_src)

        # TODO 1: Assert that out.shape matches self.x_src.shape
        self.assertEqual(out.shape, self.x_src.shape)

    def test_decoder_block_shape(self):
        decoder_block = DecoderBlock(d_model=self.d_model, heads=self.heads, d_ff=self.d_ff)
        out = decoder_block(x=self.x_tgt, memory=self.x_src)

        # TODO 2: Assert that out.shape matches self.x_tgt.shape
        self.assertEqual(out.shape, self.x_tgt.shape)

if __name__ == "__main__":
    unittest.main()