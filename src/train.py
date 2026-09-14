import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from dataset import get_synthetic_dataloader
from transformer import NanoTransformer
from utils import make_causal_mask

def get_label_smoothed_ce_loss(pad_idx: int, label_smoothing: float = 0.1) -> nn.CrossEntropyLoss:
    """
    Constructs a PyTorch CrossEntropyLoss instance configured for sequence-to-sequence training.

    Args:
        pad_idx: The integer index representing the padding token (must be ignored in loss calculation).
        label_smoothing: The float value for label smoothing regularization.

    Returns:
        An instance of nn.CrossEntropyLoss with padding ignored and label smoothing enabled.
    """
    return nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=label_smoothing)

class TransformerLRScheduler(_LRScheduler):
    """
    Learning rate scheduler matching the Attention Is All You Need paper formula.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int = 4000, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # PyTorch tracks step count via self._step_count
        step = max(1, self._step_count)
        lr_scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return [lr * lr_scale for lr in self.base_lrs]

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

    for x in dataloader:
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.to(device)

        # 1. Autoregressive split: inputs (0 to N-1), targets (1 to N)
        inputs = x[:, :-1]
        targets = x[:, 1:]

        # 2. Causal Masking (Batch, 1, Seq_Len, Seq_Len)
        mask = make_causal_mask(inputs, pad_idx)

        # 3. Forward pass
        output = model(inputs, mask=mask)

        # 4. Calculate loss
        vocab_size = output.size(-1)
        output = output.view(-1, vocab_size)
        targets = targets.reshape(-1)
        loss = loss_fn(output, targets)

        # 5. Backward pass & optimization
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

    # 2. Model
    model = NanoTransformer(
        vocab_size=VOCAB_SIZE,
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
        print(f"Epoch {epoch + 1}/{EPOCHS} | Average loss: {loss:.4f}")
