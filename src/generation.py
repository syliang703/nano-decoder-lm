import torch
import torch.nn.functional as F
from src.transformer import NanoTransformer


def sample_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Filters logits keeping only the top-k highest values, masking all others with -inf.

    Args:
        logits: Tensor of shape (batch_size, vocab_size)
        k: Int number of top candidates to keep

    Returns:
        filtered_logits: Tensor of shape (batch_size, vocab_size)
    """
    # 1. Get top-k values along vocab_size dim
    topk_values, _ = torch.topk(logits, k, dim=-1)

    # 2. Extract min value across all batches
    min_vals = topk_values[:, -1:]

    # 3. Mask everything lower than the min value with -inf
    filtered_logits = torch.where(logits < min_vals, float("-inf"), logits)
    return filtered_logits


def sample_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Filters logits using Nucleus (Top-p) sampling, keeping tokens with cumulative probability <= p.

    Args:
        logits: Tensor of shape (batch_size, vocab_size)
        p: Float cumulative probability threshold (0.0 < p <= 1.0)

    Returns:
        filtered_logits: Tensor of shape (batch_size, vocab_size)
    """
    # Edge case: Desired cumulative probability >= 1
    if p >= 1.0:
        return logits

    # 1. Take softmax to convert logits to probabilities
    probs = F.softmax(logits, dim=-1)

    # 2. Sort in descending order of probabilities
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)

    # 3. Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 4. Create mask to remove extra logits, shifting mask right by 1 to account for logit that pushed over threshold
    sorted_remove_idx = cumulative_probs > p
    sorted_remove_idx[:, 1:] = sorted_remove_idx[:, :-1].clone()
    sorted_remove_idx[:, 0] = False # Ensure first token is always kept

    # 5. Scatter removal mask back to logits tensor
    remove_idx = sorted_remove_idx.scatter(dim=-1, index=sorted_idx, src=sorted_remove_idx)

    # 6. Mask unneeded logits with -inf
    filtered_logits = logits.masked_fill(remove_idx, float("-inf"))

    return filtered_logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Applies temperature scaling, top-k, top-p filtering, and samples the next token index.

    Args:
        logits: Raw output logits for the last token position of shape (batch_size, vocab_size)
        temperature: Float scaling factor (> 0.0). If 0.0, performs greedy argmax selection.
        top_k: Int top-k candidate threshold (0 disables top-k)
        top_p: Float top-p cumulative probability threshold (1.0 disables top-p)

    Returns:
        next_token_ids: Tensor of sampled token indices of shape (batch_size, 1)
    """
    # 1. Greedy selection if non-positive temp, else scale by temp
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature

    # 2. Run top-k and top-p if enabled
    if top_k > 0:
        logits = sample_top_k(logits=logits, k=top_k)

    if top_p < 1.0:
        logits = sample_top_p(logits=logits, p=top_p)

    # 3. Take softmax to convert logits to probabilities
    probs = F.softmax(logits, dim=-1)

    next_token_ids = torch.multinomial(probs, num_samples=1)
    return next_token_ids


class Generator:
    def __init__(self, model: NanoTransformer):
        self.model = model

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressively generates new tokens using the model's stateful KV-cache.

        Args:
            prompt_ids: Input token tensor of shape (batch_size, seq_len)
            max_new_tokens: Int number of tokens to generate
            temperature: Float temperature scaling factor
            top_k: Int top-k threshold
            top_p: Float top-p threshold

        Returns:
            generated_ids: Tensor of full sequence (prompt + new tokens) of shape (batch_size, seq_len + max_new_tokens)
        """
        generated_ids = prompt_ids

        # 1. First run to initiate kv_cache and to use the entire prompt to sample first token
        logits, kv_cache = self.model(prompt_ids, use_cache=True, kv_cache = None)
        last_logits = logits[:, -1, :]
        next_token_ids = sample_next_token(logits=last_logits, temperature=temperature, top_k=top_k, top_p=top_p)
        generated_ids = torch.cat((generated_ids, next_token_ids), dim=-1)

        # 2. Run the rest of the loop starting with the first generated token as input
        for _ in range(1, max_new_tokens):
            logits, kv_cache = self.model(next_token_ids, use_cache=True, kv_cache=kv_cache)
            last_logits = logits[:, -1, :]
            next_token_ids = sample_next_token(logits=last_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            generated_ids = torch.cat((generated_ids, next_token_ids), dim=-1)

        return generated_ids