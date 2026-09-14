import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: float = 0.0,
    training: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Scaled Dot-Product Attention: Softmax(Q * K^T / sqrt(d_k)) * V

    Args:
        Q: Queries tensor of shape (batch_size, num_heads, seq_len, d_k)
        K: Keys tensor of shape (batch_size, num_heads, seq_len, d_k)
        V: Values tensor of shape (batch_size, num_heads, seq_len, d_v)
        mask: Optional tensor for masking (0 or False where attention is prevented)
        dropout: Float probability of dropout applied to attention probabilities
        training: Bool indicating training vs eval

    Returns:
        output: Tensor of shape (batch_size, num_heads, seq_len, d_v)
        attention_weights: Tensor of shape (batch_size, num_heads, seq_len, key_seq_len)
    """
    d_k = Q.size(-1)

    # 1. Compute scaled similarity scores (Q x K^T / sqrt(d_k))
    scores = Q @ torch.transpose(K, -2, -1) / (d_k ** 0.5)

    # 2. Apply optional mask (replace masked positions where mask == 0 with -1e9)
    if mask is not None:
        scores = scores.masked_fill(mask == 0.0, -1e9)

    # 3. Compute Softmax along the last dimension to get probabilities
    attention_weights = F.softmax(scores, dim=-1)

    if dropout > 0.0 and training:
        attention_weights = F.dropout(attention_weights, p=dropout)

    # 4. Compute weighted average of Values (attention_weights x V)
    output = attention_weights @ V

    return output, attention_weights

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize x, scale by gamma and shift by beta, using the mean and variance across the last dimension.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
        """
        mean = torch.mean(x, -1, keepdim=True)
        var = torch.var(x, -1, keepdim=True, unbiased=False)

        return self.gamma * (x - mean) / ((var + self.eps) ** 0.5) + self.beta

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding slice to x and apply dropout.

        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)

        Returns:
            output: Tensor of shape (batch_size, seq_length, d_model)
        """
        return self.dropout(x + self.pe[:, :x.size(1)])

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshapes input (batch_size, seq_len, d_model) to (batch_size, num_heads, seq_len, d_k)

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            output: Tensor of shape (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshapes input (batch_size, num_heads, seq_len, d_k) back to (batch_size, seq_len, d_model)

        Args:
            x: Tensor of shape (batch_size, num_heads, seq_len, d_k)

        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Implement multihead attention to input Q, K and V.

        Args:
            Q, K, V: Input tensors of shape (batch_size, seq_len, d_model)
            mask: Optional tensor for masking

        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
            attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
        """
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask, self.dropout.p, self.training)
        output = self.combine_heads(output)
        output = self.dropout(self.W_o(output))

        return output, attention_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement positionwise feedforward to output of multihead attention.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # Pre-LN Self-Attention + Residual
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=mask)
        x = x + attn_out

        # Pre-LN Feed-Forward + Residual
        x = x + self.ffn(self.norm2(x))

        return x

class NanoTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Embeddings & Positional Encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.dropout = nn.Dropout(dropout)

        # 2. Stacked Decoder Layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
            ])

        # 3. Final Pre-Output Normalization & LM Head
        self.final_norm = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 4. Weight Tying between Embeddings and LM Head
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            idx: Tensor of token indices with shape (Batch, Seq_Len)
            mask: Optional attention mask (Batch, 1, Seq_Len, Seq_Len)

        Returns:
            Logits tensor of shape (Batch, Seq_Len, Vocab_Size)
        """
        # 1. Embed tokens and add positional encodings
        x = self.token_embedding(idx) * (self.d_model ** 0.5)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 2. Pass sequentially through decoder blocks
        for layer in self.layers:
            x = layer(x, mask=mask)

        # 3. Apply final norm and project to vocabulary logits
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits