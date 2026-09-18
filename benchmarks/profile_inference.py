import time
import torch
from src.generation import Generator
from src.model import NanoTransformer

def measure_execution_time(fn, *args, **kwargs):
    """
    Helper function to accurately measure execution time across CPU and CUDA.
    Returns elapsed time in milliseconds (ms).
    """
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        output = fn(*args, **kwargs)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        start_time = time.perf_counter()
        output = fn(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000.0

    return output, elapsed_ms

import torch

def profile_memory_allocated(fn, *args, **kwargs) -> tuple:
    """
    Measures peak GPU memory allocated (in MB) during function execution.
    Should handle CPU gracefully by returning 0.0 MB if CUDA is unavailable.

    Returns:
        (output, peak_memory_mb)
    """
    # TODO: Reset peak memory stats if CUDA is available
    # TODO: Execute fn(*args, **kwargs)
    # TODO: Measure max_memory_allocated() and convert bytes to MB
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        output = fn(*args, **kwargs)
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) # Converting to MB

    else:
        output = fn(*args, **kwargs)
        peak_memory_mb = 0.0

    return output, peak_memory_mb

def profile_generation(
    model: NanoTransformer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 20,
    use_cache = True
) -> dict:
    """
    Profiles generation latency for TTFT (Time-To-First-Token) and TPOT (Time-Per-Output-Token).

    Args:
        model: Instantiated NanoTransformer model.
        prompt_ids: Tensor of input token IDs of shape (batch_size, prompt_len).
        max_new_tokens: Number of tokens to generate.
        use_cache: Whether to use KV-caching during decoding.

    Returns:
        Dict containing:
            - 'ttft_ms': Time to first token in milliseconds.
            - 'tpot_ms': Average time per output token (tokens 2..N) in milliseconds.
            - 'total_time_ms': Total time for generation pass.
    """
    generator = Generator(model=model)

    # 1. Warmup pass (un-timed) to allocate memory buffers
    _ = generator.generate(prompt_ids=prompt_ids, max_new_tokens=2, use_cache=use_cache)

    # 2. Calculate times
    _, ttft_ms = measure_execution_time(
        generator.generate,
        prompt_ids=prompt_ids,
        max_new_tokens=1,
        use_cache=use_cache
        )

    _, total_time_ms = measure_execution_time(
        generator.generate,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache
        )
    tpot_ms = (total_time_ms - ttft_ms) / (max_new_tokens - 1)

    return {'ttft_ms': ttft_ms, 'tpot_ms': tpot_ms, 'total_time_ms': total_time_ms}

def profile_uncached_vs_cached(
    model, prompt_ids: torch.Tensor, max_new_tokens: int = 20
) -> dict:
    """
    Profiles generation latency and peak VRAM usage comparing use_cache=True vs use_cache=False.

    Returns:
        Dict containing:
            - 'cached_tpot_ms': float
            - 'uncached_tpot_ms': float
            - 'speedup_factor': float (uncached / cached)
            - 'peak_vram_mb': float
    """
    generator = Generator(model=model)

    cached_timings = profile_generation(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        use_cache=True
        )
    cached_tpot_ms = cached_timings['tpot_ms']

    uncached_timings = profile_generation(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        use_cache=False
        )
    uncached_tpot_ms = uncached_timings['tpot_ms']

    speedup_factor = uncached_tpot_ms / cached_tpot_ms

    _, peak_vram_mb = profile_memory_allocated(
        generator.generate,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens
        )

    return {'cached_tpot_ms': cached_tpot_ms,
            'uncached_tpot_ms': uncached_tpot_ms,
            'speedup_factor': speedup_factor,
            'peak_vram_mb': peak_vram_mb}

if __name__ == "__main__":
    VOCAB_SIZE = 5000
    D_MODEL = 64
    NUM_HEADS = 16
    NUM_LAYERS = 16

    model = NanoTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
        )

    prompt_lengths = [16, 64, 256, 512]
    max_new_tokens = 64

    print("Prompt Len | TTFT (ms) | TPOT (ms) | Total Time (ms)")

    for prompt_length in prompt_lengths:
        prompt = torch.randint(0, VOCAB_SIZE, (1, prompt_length))
        timings = profile_generation(
            model=model,
            prompt_ids=prompt,
            max_new_tokens=max_new_tokens,
            use_cache=True
            )
        print(f"{prompt_length} | {timings['ttft_ms']:.2f} | {timings['tpot_ms']:.2f} | {timings['total_time_ms']:.2f}")