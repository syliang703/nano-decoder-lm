import time
import torch
from src.generation import Generator
from src.transformer import NanoTransformer


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


def profile_generation(
    model: NanoTransformer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 20,
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
    _ = generator.generate(prompt_ids=prompt_ids, max_new_tokens=2)

    # 2. Calculate times
    _, ttft_ms = measure_execution_time(generator.generate, prompt_ids=prompt_ids, max_new_tokens=1)
    _, total_time_ms = measure_execution_time(generator.generate, prompt_ids=prompt_ids, max_new_tokens=max_new_tokens)
    tpot_ms = (total_time_ms - ttft_ms) / (max_new_tokens - 1)

    return {'ttft_ms': ttft_ms, 'tpot_ms': tpot_ms, 'total_time_ms': total_time_ms}

if __name__ == "__main__":
    # TODO: 1. Setup model parameters (vocab_size, d_model, num_heads, num_layers) and instantiate NanoTransformer
    VOCAB_SIZE = 5000
    D_MODEL = 64
    NUM_HEADS = 16
    NUM_LAYERS = 16

    model = NanoTransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS
                            )
    # TODO: 2. Define prompt lengths to evaluate (e.g., [16, 64, 256, 512]) and max_new_tokens
    prompt_lengths = [16, 64, 256, 512]
    max_new_tokens = 64

    # TODO: 3. Print a formatted summary table header (Prompt Len | TTFT (ms) | TPOT (ms) | Total Time (ms))
    print("Prompt Len | TTFT (ms) | TPOT (ms) | Total Time (ms)")

    # TODO: 4. Loop through prompt lengths:
    #   - Generate random dummy prompt tensor of shape (1, prompt_len)
    #   - Call profile_generation()
    #   - Print formatted row metrics
    for prompt_length in prompt_lengths:
        prompt = torch.randint(0, VOCAB_SIZE, (1, prompt_length))
        timings = profile_generation(model=model, prompt_ids=prompt, max_new_tokens=max_new_tokens)
        print(f"{prompt_length} | {timings['ttft_ms']:.2f} | {timings['tpot_ms']:.2f} | {timings['total_time_ms']:.2f}")