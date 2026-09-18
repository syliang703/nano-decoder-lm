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

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps # Stabilization constant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gamma

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) FFN block.
    """
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        # Define linear projections (w1 for Swish gate, w2 for output down projection, w3 for value projection)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

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

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Adds positional encoding slice to x and apply dropout.

        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
            start_pos: Int starting position index for incremental sequence generation

        Returns:
            output: Tensor of shape (batch_size, seq_length, d_model)
        """
        seq_len = x.size(1)
        # Extract correct slice of positional encodings corresponding to current positions
        pos_slice = self.pe[:, start_pos : start_pos + seq_len]
        return self.dropout(x + pos_slice)

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
        mask: torch.Tensor = None,
        use_cache: bool = False,
        kv_cache: tuple[torch.Tensor, torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Implement multihead attention to input Q, K and V.

        Args:
            Q, K, V: Input tensors of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask tensor of size (batch_size, 1, seq_len, seq_len)
            use_cache: Bool indicating if KV caching is enabled
            kv_cache: KV cache from previous step, tuple of (K, V) with shape (batch_size, num_heads, past_seq_len, head_dim)

        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
            attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
            (K, V): If KV caching is enabled, returns cached tuple of (K, V) of shape (batch_size, num_heads, past_seq_len + seq_len, head_dim)
        """
        # 1. Linear projections
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # 2. Split heads -> shape: (batch_size, num_heads, seq_len, head_dims)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. Concat past KV cache with current token's K and V
        if kv_cache is not None:
            past_K, past_V = kv_cache
            K = torch.cat([past_K, K], dim=2)
            V = torch.cat([past_V, V], dim=2)
            mask = None # Masking not required for single token steps

        # 4. Compute attention, recombine heads and apply output projection
        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask, self.dropout.p, self.training)
        output = self.combine_heads(output)
        output = self.dropout(self.W_o(output))

        # 5. Return cached KV if enabled
        if use_cache:
            return output, attention_weights, (K, V)
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
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor = None,
            use_cache: bool = None,
            kv_cache: tuple[torch.Tensor, torch.Tensor] = None
        ):
        """
        Args:
        x: Input tensor of shape (batch_size, seq_len, d_model)
        mask: Optional attention mask of shape (batch_size, 1, seq_len, seq_len)
        use_cache: Bool indicating if KV caching is enabled
        kv_cache: Past (K, V) values for this specific layer of shape (batch_size, num_heads, past_seq_len, head_dim)

        Outputs:
        x: Transformer block output of shape(batch_size, seq_len, d_model)
        present_kv: Updated (K, V) values if KV caching is enabled of shape (batch_size, num_heads, past_seq_len + seq_len, head_dim)
        """
        norm_x = self.norm1(x)

        if use_cache:
            attn_out, _, present_kv = self.mha(norm_x, norm_x, norm_x, mask=mask, use_cache=True, kv_cache=kv_cache)
        else:
            attn_out, _ = self.mha(norm_x, norm_x, norm_x, mask=mask, use_cache=False)

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))

        if use_cache:
            return x, present_kv
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

    def _build_causal_mask(self, seq_len: int, device: torch.device):
        """
        Generates a causal mask for multi token forward passes if a mask is not passed

        Args:
            seq_len: Int indicating seq_len

        Returns:
            mask: Lower triangular causal mask of shape (1, 1, seq_len, seq_len)
        """
        return torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0).unsqueeze(0)

    def _get_start_pos(self, kv_cache):
        """
        Extracts starting position offset from the existing key cache.

        Args:
            kv_cache: List of cached (K, V) values of shape (batch_size, num_heads, past_seq_len, head_dim
            )
        Returns:
            start_pos: starting position taking into account the offset from existing key cache
        """
        if kv_cache and len(kv_cache) > 0 and kv_cache[0] is not None:
            # kv_cache[0][0] is past Key tensor of shape (batch_size, num_heads, past_seq_len, head_dim)
            return kv_cache[0][0].size(2)
        return 0

    def forward(
            self,
            idx: torch.Tensor,
            mask: torch.Tensor = None,
            use_cache: bool = False,
            kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = None
        ):
        """
        Args:
            idx: Tensor of token indices with shape (batch, seq_len)
            mask: Optional attention mask (batch_size, 1, seq_len, seq_len)
            use_cache: Bool indicating if KV caching is enabled
            kv_cache: List of num_layers (K, V) tuples of shape (batch_size, num_heads, past_seq_len, head_dim)

        Returns:
            logits: Logits tensor of shape (batch_size, seq_len, vocab_size)
            present_kv_list: If KV caching enabled, list of num_layers (K, V) tuples of size (batch_size, num_heads, past_seq_len + seq_len, head_dim)
        """
        seq_len = idx.size(1)

       # 1. Handle causal mask setup if mask not passed
        if mask is None and seq_len > 1 and not use_cache:
            mask = self._build_causal_mask(seq_len, idx.device)

        # 2. Handle positional offset
        start_pos = self._get_start_pos(kv_cache)

        # 3. xEmbeddings and positional encoding
        x = self.token_embedding(idx) * (self.d_model ** 0.5)
        x = self.pos_encoding(x, start_pos=start_pos)
        x = self.dropout(x)

        # 4. Layer Propagation
        present_kv_list = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_kv = kv_cache[i] if kv_cache is not None else None
            if use_cache:
                x, present_kv = layer(x, mask=mask, use_cache=True, kv_cache=layer_kv)
                present_kv_list.append(present_kv)
            else:
                x = layer(x, mask=mask)

        # 5. Head output
        x = self.final_norm(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, present_kv_list
        return logits