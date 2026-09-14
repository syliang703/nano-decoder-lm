import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.train import get_label_smoothed_ce_loss, TransformerLRScheduler, train_one_epoch
from src.transformer import NanoTransformer

def test_label_smoothed_ce_loss():
    """Verify label smoothing cross entropy executes and yields scalar tensor."""
    vocab_size = 10
    logits = torch.randn(2, 5, vocab_size)
    targets = torch.randint(0, vocab_size, (2, 5))

    criterion = get_label_smoothed_ce_loss(pad_idx=1, label_smoothing=0.1)
    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

    assert torch.is_tensor(loss)
    assert loss.dim() == 0  # Scalar tensor

def test_train_one_epoch():
    """Verify train_one_epoch runs successfully for one iteration with decoder-only model."""
    vocab_size = 20
    seq_len = 8
    batch_size = 4

    # Create dummy dataset of shape (num_samples, seq_len + 1)
    dummy_data = torch.randint(2, vocab_size, (16, seq_len + 1))
    dataloader = DataLoader(TensorDataset(dummy_data), batch_size=batch_size, shuffle=True)

    model = NanoTransformer(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=2,
        num_layers=1,
        d_ff=64,
        max_len=seq_len,
        dropout=0.0
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = TransformerLRScheduler(optimizer, d_model=32, warmup_steps=10)
    loss_fn = get_label_smoothed_ce_loss(pad_idx=1)
    device = torch.device("cpu")

    avg_loss = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        pad_idx=1,
        device=device
    )

    assert isinstance(avg_loss, float)
    assert avg_loss > 0.0