import pytest
import torch
from benchmarks.profile_inference import profile_generation
from src.transformer import NanoTransformer


def test_profile_generation_metrics():
    model = NanoTransformer(vocab_size=512, d_model=32, num_heads=2, num_layers=2)
    prompt = torch.tensor([[1, 25, 13],
                           [47, 51, 36]],
                           dtype=torch.long)

    timings = profile_generation(model, prompt_ids=prompt, max_new_tokens=20)

    # 1. Assert dictionary keys exist
    assert 'ttft_ms' in timings
    assert 'tpot_ms' in timings
    assert 'total_time_ms' in timings

    # 2. Assert latency values are strictly positive
    assert timings['ttft_ms'] > 0.0
    assert timings['tpot_ms'] > 0.0
    assert timings['total_time_ms'] > 0.0

    # 3. Assert total_time_ms is greater than ttft_ms
    assert timings['total_time_ms'] > timings['ttft_ms']