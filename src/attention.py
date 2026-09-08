import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Scaled Dot-Product Attention: Softmax(Q * K^T / sqrt(d_k)) * V

    Args:
        Q: Queries tensor of shape (batch_size, num_heads, seq_len, d_k)
        K: Keys tensor of shape (batch_size, num_heads, seq_len, d_k)
        V: Values tensor of shape (batch_size, num_heads, seq_len, d_v)
        mask: Optional tensor for masking (0 or False where attention is prevented)

    Returns:
        output: Tensor of shape (batch_size, num_heads, seq_len, d_v)
        attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
    """
    d_k = Q.size(-1)

    # TODO 1: Compute scaled similarity scores (Q x K^T / sqrt(d_k))
    # Hint: K needs to be transposed along its last two dimensions (-2, -1)
    scores = Q @ torch.transpose(K, -2, -1) / (d_k ** 0.5)

    # TODO 2: Apply optional mask (replace masked positions where mask == 0 with -1e9)
    if mask is not None:
        scores = scores.masked_fill(mask == 0.0, -1e9)

    # TODO 3: Compute Softmax along the last dimension to get probabilities
    attention_weights = F.softmax(scores, dim=-1)

    # TODO 4: Compute weighted average of Values (attention_weights x V)
    output = attention_weights @ V

    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # TODO 1: Instantiate linear projection layers W_q, W_k, W_v, and W_o
        # PyTorch hint: nn.Linear(in_features, out_features)
        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshapes input (batch_size, seq_len, d_model)
        to (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()

        # TODO 2: Reshape x from (batch_size, seq_len, d_model)
        # to (batch_size, seq_len, self.num_heads, self.d_k),
        # then swap dimensions 1 and 2 to get (batch_size, self.num_heads, seq_len, self.d_k)
        # Hint: x.view(...).transpose(1, 2)
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshapes input (batch_size, num_heads, seq_len, d_k)
        back to (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.size()

        # TODO 3: Swap dimensions 1 and 2 back, make memory contiguous,
        # and reshape back to (batch_size, seq_len, self.d_model)
        # Hint: x.transpose(1, 2).contiguous().view(...)
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q, K, V: Input tensors of shape (batch_size, seq_len, d_model)
            mask: Optional tensor for masking
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
            attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
        """
        # TODO 4: Linear projections for Q, K, V
        # Hint: Q = self.W_q(Q), etc.
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # TODO 5: Split heads for Q, K, V using self.split_heads()
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # TODO 6: Pass split Q, K, V, and mask into scaled_dot_product_attention()
        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        # TODO 7: Combine output heads back using self.combine_heads()
        output = self.combine_heads(output)

        # TODO 8: Apply output projection self.W_o to the combined output
        output = self.W_o(output)

        return output, attention_weights

if __name__ == "__main__":
    # Target Apple GPU (MPS) if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running execution on device: {device}")

    batch_size, num_heads, seq_len, d_k = 1, 1, 4, 8
    torch.manual_seed(42)

    Q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
    K = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
    V = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)

    try:
        output, weights = scaled_dot_product_attention(Q, K, V)

        assert output is not None, "Output is None! Complete the TODOs."
        assert output.shape == (batch_size, num_heads, seq_len, d_k), f"Incorrect output shape: {output.shape}"
        assert weights.shape == (batch_size, num_heads, seq_len, seq_len), f"Incorrect weights shape: {weights.shape}"

        row_sums = weights.sum(dim=-1).squeeze().cpu()
        assert torch.allclose(row_sums, torch.ones_like(row_sums)), "Softmax rows do not sum to 1.0!"

        print("\nAll assertions passed successfully!")
        print("Attention Weights shape:", weights.shape)
        print("Output shape:", output.shape)

    except Exception as e:
        print(f"\nExecution error: {e}")

