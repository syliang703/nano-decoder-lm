import torch
import torch.nn as nn
from attention import MultiHeadAttention
from modules import PositionwiseFeedForward, LayerNorm

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Initialize sub-layers
        self.self_attn = MultiHeadAttention(d_model, heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # TODO 1: Initialize two LayerNorm instances (one for self-attn residual, one for FFN residual)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)

        # TODO 2: First sub-layer: Multi-Head Self-Attention + Residual + LayerNorm
        # 1. Pass x through self_attn (Q=x, K=x, V=x, mask=mask)
        # 2. Apply dropout1 to attention output
        # 3. Add residual connection to x
        # 4. Apply norm1
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))


        # TODO 3: Second sub-layer: Feed-Forward Network + Residual + LayerNorm
        # 1. Pass x through feed_forward
        # 2. Apply dropout2
        # 3. Add residual connection
        # 4. Apply norm2
        ffn_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads)
        self.cross_attn = MultiHeadAttention(d_model, heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # TODO 4: Initialize three LayerNorm instances (one per sub-layer) using d_model
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        # x shape: (batch_size, tgt_seq_len, d_model)
        # memory shape (encoder output): (batch_size, src_seq_len, d_model)

        # TODO 5: Sub-layer 1: Masked Self-Attention + Residual + LayerNorm
        # Pass x as Q, K, V with tgt_mask, apply dropout1, add residual x, apply norm1
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        # TODO 6: Sub-layer 2: Cross-Attention + Residual + LayerNorm
        # Pass Q=x, K=memory, V=memory with src_mask, apply dropout2, add residual x, apply norm2
        cross_attn_out, _ = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        # Sub-layer 3: Feed-Forward + Residual + LayerNorm
        ffn_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ffn_out))

        return x