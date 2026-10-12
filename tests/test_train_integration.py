import os
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from src.checkpoint import save_checkpoint, load_checkpoint
from src.dataset import PretrainedDataset
from src.model import NanoTransformer
from src.train import configure_optimizers, train_step


@pytest.fixture
def tiny_training_bin(tmp_path):
    """Fixture creating a tiny token binary file for fast integration smoke tests."""
    bin_path = os.path.join(tmp_path, "tiny_tokens.bin")
    # Generate 500 deterministic tokens
    tokens = np.random.randint(0, 100, size=500, dtype=np.uint16)
    tokens.tofile(bin_path)
    return str(bin_path)


def test_end_to_end_training_smoke(tiny_training_bin, tmp_path):
    """Verifies that a multi-step training run executes, saves checkpoints, and logs metrics successfully."""
    dataset = PretrainedDataset(bin_path=tiny_training_bin, block_size=16)
    dataloader = DataLoader(dataset, batch_size=2)

    # Run dedicated integration test runner
    model = NanoTransformer(vocab_size=500,
                            d_model=32,
                            num_heads=2,
                            num_layers=2,
                            d_ff=32,
                            max_len=64,
                            dropout=0.1
                            )

    optimizer = configure_optimizers(model=model,
                                    weight_decay=0.1,
                                    learning_rate=1e-3,
                                    betas=(0.5, 0.5),
                                    device_type='cpu'
                                    )

    losses = []
    for epoch in range(5):
        for x, y in dataloader:
            loss = train_step(model=model,
                              optimizer=optimizer,
                              x=x,
                              y=y,
                              grad_clip=1.0,
                              ptdtype=torch.bfloat16,
                              device_type='cpu'
                              )

        losses.append(loss)
        print(f'Epoch: {epoch} | Loss: {loss:.4f}')

        save_checkpoint(model=model,
                        optimizer=optimizer,
                        iteration=epoch,
                        val_loss=loss,
                        filepath=os.path.join(tmp_path, f"checkpoint_{epoch}.pt")
                        )

    # Assert that checkpoint files are created on disk and loss values are recorded correctly
    checkpoint3_path = str(tmp_path / "checkpoint_3.pt")
    assert os.path.exists(checkpoint3_path)

    checkpoint3 = load_checkpoint(filepath=checkpoint3_path, model = model)
    assert 'loss' in checkpoint3
    assert losses[3] == checkpoint3['loss']