import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import get_synthetic_dataloader
from transformer import Transformer
from scheduler import TransformerLRScheduler
from loss import get_label_smoothed_ce_loss
from utils import make_src_mask, make_tgt_mask

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    loss_fn: nn.Module,
    pad_idx: int,
    device: torch.device
) -> float:
    """
    Runs one training epoch over the dataloader.

    Returns:
        Average loss over all batches in the epoch.
    """
    model.train()
    total_loss = 0.0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        # Slice target tensor for target input and target labels (teacher forcing)
        # TODO 1: Extract tgt_input (all columns except the last)
        # TODO 2: Extract tgt_label (all columns except the first)
        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]

        # Generate masks
        # TODO 3: Create src_mask using make_src_mask(src, pad_idx)
        # TODO 4: Create tgt_mask using make_tgt_mask(tgt_input, pad_idx)
        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx)

        # Forward pass
        # TODO 5: Pass src, tgt_input, src_mask, tgt_mask into model(...)
        output = model(src, tgt_input, src_mask, tgt_mask)

        # Calculate loss
        # TODO 6: Flatten logits to (N, vocab_size) and tgt_label to (N,)
        # TODO 7: Compute scalar loss using loss_fn
        vocab_size = output.size(-1)
        output = output.view(-1, vocab_size)
        tgt_label = tgt_label.reshape(-1)
        loss = loss_fn(output, tgt_label)

        # Backward pass & optimization
        # TODO 8: Zero gradients, execute backward pass, step optimizer, step scheduler
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

if __name__ == "__main__":
    # Hyperparameters
    VOCAB_SIZE = 100
    SEQ_LEN = 12
    BATCH_SIZE = 32
    NUM_SAMPLES = 640
    D_MODEL = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2
    D_FF = 256
    DROPOUT = 0.1
    EPOCHS = 40
    PAD_IDX = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. DataLoader
    dataloader = get_synthetic_dataloader(
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        num_samples=NUM_SAMPLES
        )

    # 2. Model (Import your complete Transformer class from src.transformer)
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=SEQ_LEN,
        dropout=DROPOUT
    )

    # 3. Optimizer, Scheduler, Loss Function
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerLRScheduler(optimizer=optimizer, d_model=D_MODEL, warmup_steps=400)
    loss_fn = get_label_smoothed_ce_loss(pad_idx=PAD_IDX)

    # 4. Training Loop across EPOCHS
    for epoch in range(EPOCHS):
        loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            pad_idx=PAD_IDX,
            device=device
            )
        print(f"Average loss: {loss}")
