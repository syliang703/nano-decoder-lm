import pytest
import torch
from src.dataset import get_synthetic_dataloader
from src.generation import Generator, sample_next_token, sample_top_k, sample_top_p
from src.model import NanoTransformer

@pytest.fixture
def dataloader_params():
    return {
        "vocab_size": 50,
        "seq_len": 10,
        "batch_size": 4,
        "num_samples": 16,
    }

def test_dataloader_batch_shape(dataloader_params):
    dataloader = get_synthetic_dataloader(
        vocab_size=dataloader_params["vocab_size"],
        seq_len=dataloader_params["seq_len"],
        batch_size=dataloader_params["batch_size"],
        num_samples=dataloader_params["num_samples"],
    )

    # 1. Get the first batch from the dataloader
    batch = next(iter(dataloader))

    # 2. Assert batch shape is (batch_size, seq_len + 1)
    expected_shape = (dataloader_params["batch_size"], dataloader_params["seq_len"] + 1)
    assert batch.shape == expected_shape

    # 3. Assert tensor data type is integer (torch.long or torch.int64)
    assert batch.dtype == torch.long

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
    generator = Generator(model=transformer)

    prompt = torch.tensor([[3, 2, 1],
                           [7, 9, 8]])

    output = generator.generate(prompt_ids=prompt, max_new_tokens=1, temperature=1, top_k=2, top_p=0.5)

    assert output.shape == (prompt.shape[0], prompt.shape[1] + 1)
    assert torch.equal(output[:, :prompt.size(1)], prompt)