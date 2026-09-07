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
        scores.masked_fill_(~mask, -1e9)

    # TODO 3: Compute Softmax along the last dimension to get probabilities
    attention_weights = F.softmax(scores, dim=-1)

    # TODO 4: Compute weighted average of Values (attention_weights x V)
    output = attention_weights @ V

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