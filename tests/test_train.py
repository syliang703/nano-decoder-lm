import math
import pytest
import torch
import torch.nn as nn
from src.model import NanoTransformer
from src.train import configure_optimizers, get_lr, train_step


@pytest.fixture
def dummy_model_and_data():
    """Fixture providing a lightweight NanoTransformer instance and dummy token tensors."""
    vocab_size = 100
    seq_len = 16

    model = NanoTransformer(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        max_len=128,
        dropout=0.0,
    )

    x = torch.randint(0, vocab_size, (2, seq_len))
    y = torch.randint(0, vocab_size, (2, seq_len))
    return model, x, y


def test_optimizer_parameter_groups(dummy_model_and_data):
    """Verifies weight decay separation between 2D weights and 1D biases/norms."""
    model, _, _ = dummy_model_and_data
    optimizer = configure_optimizers(
        model,
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
    )

    # 1. Assert optimizer has exactly 2 parameter groups
    assert len(optimizer.param_groups) == 2

    # 2. Assert group 0 has weight_decay == 0.1
    assert optimizer.param_groups[0]["weight_decay"] == 0.1

    # 3. Assert group 1 has weight_decay == 0.0
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_cosine_learning_rate_schedule():
    """Verifies warmup ramp and cosine decay LR calculation."""
    max_lr = 1e-3
    min_lr = 1e-4
    warmup_iters = 10
    lr_decay_iters = 100

    # 1. Test step 0
    assert math.isclose(get_lr(0,
                               learning_rate=max_lr,
                               min_lr=min_lr,
                               warmup_iters=warmup_iters,
                               lr_decay_iters=lr_decay_iters
                               ),
                               max_lr / warmup_iters)
    # 2. Test warmup peak
    assert math.isclose(get_lr(warmup_iters,
                               learning_rate=max_lr,
                               min_lr=min_lr,
                               warmup_iters=warmup_iters,
                               lr_decay_iters=lr_decay_iters
                               ),
                               max_lr)
    # 3. Test post-decay floor
    assert get_lr(150,
                  learning_rate=max_lr,
                  min_lr=min_lr,
                  warmup_iters=warmup_iters,
                  lr_decay_iters=lr_decay_iters
                  ) == min_lr


def test_train_step_weight_update_and_loss(dummy_model_and_data):
    """Verifies model parameters update post train_step and loss returns valid float."""
    model, x, y = dummy_model_and_data
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Capture initial weights of first parameter tensor
    initial_param = next(model.parameters()).clone().detach()

    # Perform single training step
    loss_val = train_step(
        model=model,
        optimizer=optimizer,
        x=x,
        y=y,
        grad_clip=1.0,
        device_type="cpu",
        ptdtype=torch.bfloat16,
    )

    # 1. Assert loss_val is an instance of float and > 0.0
    assert isinstance(loss_val, float)
    assert loss_val > 0.0
    # 2. Assert initial_param and next(model.parameters()) are not equal
    assert not torch.equal(initial_param, next(model.parameters()))