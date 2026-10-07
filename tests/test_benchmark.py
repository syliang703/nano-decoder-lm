import os
import pytest
import torch
from benchmarks.profile_inference import (
    export_benchmark_results,
    profile_generation,
    profile_memory_allocated,
    profile_uncached_vs_cached,
    run_benchmark_sweep
)
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
    # Assert speedup factor > 1.0 (cached is faster than uncached)
    VOCAB_SIZE = 4000
    model = NanoTransformer(vocab_size=VOCAB_SIZE)
    prompt = torch.randint(0, VOCAB_SIZE, (1, 100))
    results = profile_uncached_vs_cached(model=model, prompt_ids=prompt, max_new_tokens=200)

    assert 'cached_tpot_ms' in results
    assert 'uncached_tpot_ms' in results
    assert 'speedup_factor' in results
    assert 'peak_vram_mb' in results
    assert results['speedup_factor'] > 1.0

def test_profile_memory_allocated_cpu_fallback():
    # TODO: Define dummy function returning a tensor
    # TODO: Call profile_memory_allocated(dummy_fn)
    # TODO: Assert peak_memory_mb is float (0.0 if non-CUDA)
    foo = lambda: torch.tensor([1.0, 2.0, 3.0, 4.0])
    _, peak_memory_allocated = profile_memory_allocated(foo)
    if torch.cuda.is_available():
        assert isinstance(peak_memory_allocated, float)
    else:
        assert peak_memory_allocated == 0.0

def test_run_benchmark_sweep_metrics():
    # TODO: Instantiate tiny NanoTransformer model
    # TODO: Call run_benchmark_sweep with short seq_lengths=[16, 32]
    # TODO: Assert returned list has length equal to input seq_lengths
    # TODO: Verify expected metric keys exist in each result dictionary
    model = NanoTransformer(vocab_size=4000)
    seq_lengths = [16, 32]
    results = run_benchmark_sweep(model=model, seq_lengths=seq_lengths)
    assert len(results) == len(seq_lengths)
    assert 'seq_len' in results[0]
    assert 'ttft_ms' in results[0]
    assert 'cached_tpot_ms' in results[0]
    assert 'uncached_tpot_ms' in results[0]
    assert 'speedup_factor' in results[0]
    assert 'peak_vram_mb' in results[0]

def test_export_benchmark_results(tmp_path):
    # TODO: Create dummy benchmark results list of dicts
    # TODO: Call export_benchmark_results with output_dir=str(tmp_path)
    # TODO: Assert both JSON and CSV files are created and non-empty
    results = [{
                "seq_len": 16,
                "ttft_ms":  1.2,
                "cached_tpot_ms": 5.4,
                "uncached_tpot_ms": 4.2,
                "speedup_factor": 1.2,
                "peak_vram_mb": 0.0001
            }]

    output_dir = str(tmp_path)
    json_path, csv_path = export_benchmark_results(results=results, output_dir=output_dir)
    assert os.path.exists(json_path)
    assert os.path.getsize(json_path) > 0.0
    assert os.path.exists(csv_path)
    assert os.path.getsize(csv_path) > 0.0