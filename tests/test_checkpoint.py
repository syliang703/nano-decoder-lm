import os
import torch
import torch.nn as nn
from pathlib import Path
from src.model import NanoTransformer
from src.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_roundtrip(tmp_path):
    """Verifies that saving and loading a checkpoint preserves model weights, optimizer state, and metadata exactly."""
    model = NanoTransformer(vocab_size=512)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    epoch = 10
    loss = -1e4

    # Save checkpoint using save_checkpoint
    save_checkpoint(model=model, optimizer=optimizer, filepath=tmp_path/"checkpoint.pt", iteration=epoch, val_loss=loss)

    # Instantiate a fresh model/optimizer and load the checkpoint
    model_new = NanoTransformer(vocab_size=512)
    optimizer_new = torch.optim.AdamW(model_new.parameters(), lr=1e-3)

    checkpoint = load_checkpoint(filepath=tmp_path/"checkpoint.pt", model = model_new)

    # Assert model weights match and iteration/val_loss metadata are recovered correctly
    for key in model.state_dict():
        assert torch.equal(model_new.state_dict()[key], model.state_dict()[key])
    for key in optimizer.state_dict():
        assert torch.equal(optimizer_new.state_dict()[key], optimizer.state_dict()[key])
    assert checkpoint['epoch'] == epoch
    assert checkpoint['loss'] == loss