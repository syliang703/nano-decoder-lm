import argparse
import torch
from src.checkpoint import load_checkpoint
from src.model import NanoTransformer
from src.tokenizer import Tokenizer
from src.generation import Generator


def main():
    parser = argparse.ArgumentParser(description="Generate text using NanoTransformer checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_4999.pt", help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time,", help="Input prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering parameter")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p nucleus filtering parameter")
    parser.add_argument("--device", type=str, default="mps", help="Device (cpu, mps, cuda)")
    parser.add_argument("--repetition_penalty", type=float, default=1.12, help="Repetition penalty factor")

    args = parser.parse_args()

    # 1. Setup target device (handling fallback if mps/cuda is unavailable)
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # 2. Load model checkpoint and state dict
    model, _ = load_checkpoint(filepath=args.checkpoint,
                               model=None,
                               optimizer=None,
                               device=device
                               )

    model.eval()

    # 3. Initialize Tokenizer and encode prompt into tensor format
    tokenizer = Tokenizer(model_name="gpt2")
    eos_id = getattr(tokenizer, "eos_id", None)

    prompt_ids = tokenizer.encode(text=args.prompt, add_bos=False, add_eos=False)
    input_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # 4. Instantiate Generator class from src.generation
    generator = Generator(model=model,
                          temperature=args.temperature,
                          top_k=args.top_k,
                          top_p=args.top_p
                          )

    # 5. Run generation, decode the output token stream, and print the result
    output_tokens = generator.generate(prompt_ids=input_tokens,
                                       max_new_tokens=args.max_new_tokens,
                                       eos_token_id=eos_id
                                       )

    output_text = tokenizer.decode(ids=output_tokens[0])
    print(output_text)


if __name__ == "__main__":
    main()