import pytest
import torch
from benchmarks.profile_inference import profile_generation, profile_uncached_vs_cached
from src.model import NanoTransformer


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

def test_uncached_vs_cached_metrics():
    # TODO: Instantiate profile_uncached_vs_cached helper
    # TODO: Assert speedup factor > 1.0 (cached is faster than uncached)
    VOCAB_SIZE = 4000
    model = NanoTransformer(vocab_size=VOCAB_SIZE)
    prompt = torch.randint(0, VOCAB_SIZE, (1, 100))
    results = profile_uncached_vs_cached(model=model, prompt_ids=prompt, max_new_tokens=200)

    assert 'cached_tpot_ms' in results
    assert 'uncached_tpot_ms' in results
    assert 'speedup_factor' in results
    assert 'peak_vram_mb' in results
    assert results['speedup_factor'] > 1.0