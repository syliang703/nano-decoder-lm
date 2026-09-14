import math
import torch
import torch.nn as nn
from blocks import EncoderBlock, DecoderBlock
from embedding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        # TODO 1: Initialize embeddings (src & tgt using nn.Embedding), positional encoding,
        # encoder layers (nn.ModuleList of EncoderBlocks),
        # decoder layers (nn.ModuleList of DecoderBlocks),
        # and the output linear generator head (nn.Linear).
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for i in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for i in range (num_layers)])

        self.output = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # TODO 2: Embed src, scale by math.sqrt(self.d_model), apply pos_encoding,
        # and pass sequentially through each block in self.encoder_layers.
        encoded_src = self.src_embedding(src) * math.sqrt(self.d_model)
        encoded_src = self.pos_encoding(encoded_src)
        for layer in self.encoder_layers:
            encoded_src = layer(encoded_src, src_mask)
        return encoded_src

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        # TODO 3: Embed tgt, scale by math.sqrt(self.d_model), apply pos_encoding,
        # and pass sequentially with memory through each block in self.decoder_layers.
        encoded_tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        encoded_tgt = self.pos_encoding(encoded_tgt)
        for layer in self.decoder_layers:
            encoded_tgt =layer(encoded_tgt, memory, src_mask, tgt_mask)
        return encoded_tgt

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        # TODO 4: Connect encode -> decode -> generator head to return target logits.
        encoded_src = self.encode(src, src_mask)
        encoded_tgt = self.decode(tgt, encoded_src, src_mask, tgt_mask)
        return self.output(encoded_tgt)