import pytest
import torch
from src.generation import Generator, sample_next_token, sample_top_k, sample_top_p, apply_repetition_penalty
from src.model import NanoTransformer

def test_greedy_sampling():
    # Check if greedy sampling works if temp is <= 0
    logits = torch.tensor([[3.0, 2.0, 1.0],
                           [7.0, 9.0, 8.0]])

    next_token_ids = sample_next_token(logits=logits, temperature=-1.0)
    expected = torch.tensor([[0], [1]])
    assert torch.equal(next_token_ids, expected)


def test_top_k_filtering():
    logits = torch.tensor([[3.0, 2.0, 1.0],
                           [7.0, 9.0, 8.0]])
    filtered_logits = sample_top_k(logits=logits, k=2)
    expected = torch.tensor([[3.0, 2.0, float("-inf")],
                             [float("-inf"), 9.0, 8.0]])
    assert torch.allclose(filtered_logits, expected)


def test_top_p_filtering():
    logits = torch.tensor([[-3.0, 2.0, 1.0],
                           [-7.0, 9.0, -8.0]])
    filtered_logits = sample_top_p(logits=logits, p=0.9)
    expected = torch.tensor([[float("-inf"), 2.0, 1.0],
                             [float("-inf"), 9.0, float("-inf")]])
    assert torch.allclose(filtered_logits, expected)


def test_generator_kv_cache_integration():
    transformer = NanoTransformer(vocab_size=10)
    generator = Generator(model=transformer, temperature=1, top_k=2, top_p=0.5)

    prompt = torch.tensor([[3, 2, 1],
                           [7, 9, 8]])

    output = generator.generate(prompt_ids=prompt, max_new_tokens=1)

    assert output.shape == (prompt.shape[0], prompt.shape[1] + 1)
    assert torch.equal(output[:, :prompt.size(1)], prompt)


def test_apply_repetition_penalty():
    # Setup mock logits (batch_size=1, vocab_size=5)
    # Token IDs 1 and 3 have appeared in the sequence history
    logits = torch.tensor([[2.0, 5.0, 1.0, 4.0, 0.5]])
    input_ids = torch.tensor([[1, 3, 1]]) # token 1 and 3 are present
    penalty = 1.5

    adjusted_logits = apply_repetition_penalty(logits, input_ids, penalty)

    assert adjusted_logits[0, 1].item() < 5.0
    assert adjusted_logits[0, 3].item() < 4.0
    assert adjusted_logits[0, 0].item() == 2.0
    assert adjusted_logits[0, 2].item() == 1.0
    assert adjusted_logits[0, 4].item() == 0.5