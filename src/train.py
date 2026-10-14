import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.checkpoint import save_checkpoint, load_checkpoint
from src.dataset import PretrainedDataset
from src.generation import Generator
from src.model import NanoTransformer


def configure_optimizers(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    device_type: str,
) -> torch.optim.AdamW:
    """Configures AdamW optimizer with weight decay applied exclusively to 2D weight matrices.

    1D parameters (biases, LayerNorm scales/biases, embedding parameters) have weight decay set to 0.0.
    """
    decay = []
    no_decay = []

    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    params = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]

    optimizer = torch.optim.AdamW(params=params,
                                  lr=learning_rate,
                                  betas=betas,
                                  )

    return optimizer


def get_lr(
    it: int,
    learning_rate: float,
    min_lr: float,
    warmup_iters: int,
    lr_decay_iters: int,
) -> float:
    """Calculates learning rate at iteration `it` using linear warmup followed by cosine decay."""
    # 1. Linear warmup if it < warmup_iters
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters

    # 2. If it > lr_decay_iters: return min_lr
    if it > lr_decay_iters:
        return min_lr

    # 3. Cosine decay between warmup_iters and lr_decay_iters:
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    grad_clip: float,
    device_type: str
) -> float:
    """Executes a single forward, backward, gradient clipping, and optimizer step under AMP."""
    optimizer.zero_grad(set_to_none=True)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    return loss.item()

def train_model():
    """Main training orchestrator for nano-decoder-lm."""
    # 1. Initialize configuration, model, dataloaders, optimizer, and AMP scaler/dtypes
    vocab_size = 50000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_len = 5000
    dropout = 0.1

    weight_decay = 0.5
    learning_rate = 1e-3
    betas = (0.7, 0.7)
    device_type = 'cpu'
    ptdtype = torch.float32

    model = NanoTransformer(vocab_size=vocab_size,
                            d_model=d_model,
                            num_heads=num_heads,
                            num_layers=num_layers,
                            d_ff=d_ff,
                            max_len=max_len,
                            dropout=dropout
                            )

    optimizer = configure_optimizers(model=model,
                                     weight_decay=weight_decay,
                                     learning_rate=learning_rate,
                                     betas=betas,
                                     device_type=device_type
                                     )

    train_data = PretrainedDataset(bin_path="training_data.bin",
                                   block_size=64
                                   )

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

    # 2. Run training loop over iterations, saving checkpoints periodically
    epochs = 100

    for epoch in range(epochs):
        for x, y in train_dataloader:
            loss = train_step(model=model,
                            optimizer=optimizer,
                            x=x,
                            y=y,
                            grad_clip=1.0,
                            ptdtype=ptdtype
                            )

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")
            save_checkpoint(model=model,
                            optimizer=optimizer,
                            iteration=epoch,
                            val_loss=loss,
                            filepath=f"checkpoint_{epoch}.pt")


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    eval_dataloader: DataLoader,
    device_type: str,
    max_eval_iters: int = 50,
) -> float:
    """Computes average cross-entropy loss over validation/evaluation batches without gradient updates."""
    model.eval()
    total_loss = 0.0
    epochs = 0

    for x, y in eval_dataloader:
        if epochs >= max_eval_iters:
            break

        x, y = x.to(device_type), y.to(device_type)

        output = model(x)
        loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1))

        total_loss += loss.item()
        epochs += 1

    model.train()
    return total_loss / max(epochs, 1)


def generate_sample_text(
    model: nn.Module,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: str
) -> torch.Tensor:
    """Generates text tokens from a prompt using the model's sampling engine."""
    model.eval()
    generator = Generator(model=model, temperature=temperature, top_k=top_k, top_p=top_p)

    with torch.no_grad():
        output_tokens = generator.generate(prompt_ids=prompt_tokens, max_new_tokens=max_new_tokens)

    model.train()
    return output_tokens


if __name__ == "__main__":
    train_model()