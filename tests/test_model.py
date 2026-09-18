import pytest
import torch
from src.model import RMSNorm, SwiGLU, MultiHeadAttention, NanoTransformer
from src.utils import make_causal_mask


def test_rmsnorm_forward_shape_and_scale():
    # TODO: Instantiate RMSNorm(dim=64)
    # TODO: Create dummy tensor x of shape (2, 16, 64)
    # TODO: Pass x through RMSNorm
    # TODO: Assert output shape == (2, 16, 64)
    norm = RMSNorm(dim=64)
    x = torch.rand((2, 16, 64))
    output = norm(x)
    assert output.shape == x.shape

def test_swiglu_forward_shape():
    # TODO: Instantiate SwiGLU(d_model=64, hidden_dim=256)
    # TODO: Create dummy tensor x of shape (2, 16, 64)
    # TODO: Pass x through SwiGLU
    # TODO: Assert output shape == (2, 16, 64)
    swiglu = SwiGLU(d_model=64, hidden_dim=256)
    x = torch.rand((2, 16, 64))
    output = swiglu(x)
    assert output.shape == x.shape

def test_nano_transformer_forward_shape():
    """Verify that input token indices map to (Batch, Seq_Len, Vocab_Size) logits."""
    batch_size = 2
    seq_len = 8
    vocab_size = 100
    d_model = 64

    # Use correct parameter names (num_heads, num_layers) matching NanoTransformer
    model = NanoTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=2,
        num_layers=2
    )
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(idx)
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Expected shape {(batch_size, seq_len, vocab_size)}, got {logits.shape}"

def test_causal_mask_prevents_future_attention():
    """Ensure causal mask correctly masks out future tokens."""
    seq_len = 4
    dummy_input = torch.zeros((1, seq_len), dtype=torch.long)
    mask = make_causal_mask(dummy_input, pad_idx=1)

    # Upper triangle (excluding diagonal) should be False indicating masked positions
    assert not mask[0, 0, 0, 1]
    # Diagonal and lower triangle should be True indicating unmasked positions
    assert mask[0, 0, 0, 0]

def test_kv_cache_shapes():
    """Ensure KV cache grows by 1 sequence dimension on each step during autoregressive generation."""
    d_model = 64
    num_heads = 2
    batch_size = 2
    head_dim = d_model // num_heads

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    mha.eval()

    # Step 1: Pass first token
    x1 = torch.randn(batch_size, 1, d_model)
    _, _, kv_cache1 = mha(x1, x1, x1, use_cache=True)
    assert kv_cache1[0].shape == (batch_size, num_heads, 1, head_dim)

    # Step 2: Pass second token with past kv_cache1
    x2 = torch.randn(batch_size, 1, d_model)
    _, _, kv_cache2 = mha(x2, x2, x2, use_cache=True, kv_cache=kv_cache1)
    assert kv_cache2[0].shape == (batch_size, num_heads, 2, head_dim)

def test_kv_cache_equivalence():
    """Verify generated outputs with KV cache match standard forward pass over full sequence."""
    d_model = 64
    num_heads = 2
    batch_size = 1
    seq_len = 3

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    mha.eval()

    x_full = torch.randn(batch_size, seq_len, d_model)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        # Compute full forward pass without cache
        out_full, _ = mha(x_full, x_full, x_full, causal_mask)

        # Step token-by-token with cache
        out_cached_list = []
        past_kv = None

        for t in range(seq_len):
            x = x_full[:, t:t+1, :]
            output, _, past_kv = mha(x, x, x, use_cache=True, kv_cache=past_kv)
            out_cached_list.append(output)

        out_cached = torch.cat(out_cached_list, dim=1)

    assert torch.allclose(out_full, out_cached, atol=1e-5)

def test_model_kv_cache_generation():
    """Verify full NanoTransformer generates identical outputs with and without KV cache."""
    vocab_size = 100
    d_model = 64
    num_heads = 2
    num_layers = 2
    max_seq_len = 32

    model = NanoTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_len=max_seq_len
    )
    model.eval()

    input_ids = torch.randint(0, vocab_size, (1, 4))  # Batch=1, Seq=4

    with torch.no_grad():
        logits_full = model(input_ids)

        past_kv_list = None
        logits_cached_list = []

        for t in range(input_ids.size(1)):
            single_token = input_ids[:, t:t+1]
            logits_step, past_kv_list = model(single_token, use_cache=True, kv_cache=past_kv_list)

            logits_cached_list.append(logits_step)

        cached_logits = torch.cat(logits_cached_list, dim=1)

        assert torch.allclose(logits_full, cached_logits, atol=1e-4)