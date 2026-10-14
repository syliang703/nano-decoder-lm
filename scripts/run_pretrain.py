import numpy as np
import math
import os
import torch
from torch.utils.data import DataLoader
from src.checkpoint import save_checkpoint, load_checkpoint
from src.dataset import PretrainedDataset
from src.model import NanoTransformer
from src.train import configure_optimizers, train_step, estimate_loss, generate_sample_text
from src.tokenizer import Tokenizer


def main():
    # 1. Hyperparameters & Configuration
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 512
    max_len = 256
    dropout = 0.1

    batch_size = 32
    block_size = 256
    learning_rate = 1e-4
    weight_decay = 0.1
    betas = (0.9, 0.95)
    grad_clip = 0.5
    max_iters = 5000
    eval_interval = 200

    temperature = 0.8
    top_k = 50
    top_p = 0.9

    global_step = 0

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    device_type = device

    ptdtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    print(f"Using device: {device} | AMP dtype: {ptdtype}")

    # 2. Data Loading
    # TODO: Initialize your tokenizer and instantiate PretrainedDataset / DataLoader for train.bin and val.bin
    tokenizer = Tokenizer(model_name="gpt2")
    vocab_size = tokenizer.vocab_size

    train_data = PretrainedDataset(bin_path="data/processed/train.bin", block_size=block_size, dtype=np.uint16)
    val_data = PretrainedDataset(bin_path="data/processed/val.bin", block_size=block_size, dtype=np.uint16)

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

    # 3. Model & Optimizer Initialization
    # TODO: Instantiate NanoTransformer and configure optimizers using configure_optimizers
    model = NanoTransformer(vocab_size=vocab_size,
                            d_model=d_model,
                            num_heads=num_heads,
                            num_layers=num_layers,
                            d_ff=d_ff,
                            max_len=max_len,
                            dropout=dropout
                            ).to(device)
    model.to(device)

    optimizer = configure_optimizers(model=model,
                                     weight_decay=weight_decay,
                                     learning_rate=learning_rate,
                                     betas=betas,
                                     device_type=device_type
                                     )

    checkpoint_path = "checkpoints/checkpoint_0.pt"

    if os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint(filepath=checkpoint_path,
                                     model=model,
                                     optimizer=optimizer,
                                     device=device
                                     )

        global_step = checkpoint['step']
        print(f"Resuming training from checkpoint {global_step}")

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        global_step += 1

    # 4. Training Loop
    # TODO: Implement training iterations, periodic loss evaluation, text generation sampling, and checkpoint saving
    sample_input = "There once was a man"
    print(f"Sample input: {sample_input}")
    sample_input_tokens = torch.tensor(tokenizer.encode(text=sample_input),
                                       dtype=torch.long,
                                       device=device
                                       ).unsqueeze(0)

    model.train()

    while global_step < max_iters:
        for x, y in train_dataloader:
            if global_step >= max_iters:
                break

            x, y = x.to(device), y.to(device)
            loss = train_step(model=model,
                              optimizer=optimizer,
                              x=x,
                              y=y,
                              grad_clip=grad_clip,
                              device_type=device_type,
                              ptdtype=ptdtype
                              )

            if math.isnan(loss) or math.isinf(loss):
                print(f"Warning: Encountered NaN/Inf loss at step {global_step}. Resetting to previous checkpoint.")

                last_checkpoint = global_step - (global_step % eval_interval) - 1

                if last_checkpoint < 0:
                    print(f"NaN/Inf loss occurred on {global_step} step before first checkpoint. Terminating. ")

                checkpoint_path = f"checkpoints/checkpoint_{last_checkpoint}.pt"

                if os.path.exists(checkpoint_path):
                    checkpoint = load_checkpoint(filepath=checkpoint_path,
                                                    model=model,
                                                    optimizer=optimizer,
                                                    device=device
                                                    )

                    global_step = checkpoint['step']
                    print(f"Resuming training from checkpoint {global_step}")

                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                    print(f"Reduced learning rate to {param_group['lr']}")

            if (global_step + 1) % eval_interval == 0:
                model.eval()
                loss_eval = estimate_loss(model=model,
                                        eval_dataloader=val_dataloader,
                                        device_type=device_type,
                                        ptdtype=ptdtype
                                        )
                model.train()

                print(f"Step: {global_step} | Train Loss: {loss:.4f} | Val Loss: {loss_eval:.4f}")

                sample_output_tokens = generate_sample_text(model=model,
                                                    prompt_tokens=sample_input_tokens,
                                                    max_new_tokens=30,
                                                    temperature=temperature,
                                                    top_k=top_k,
                                                    top_p=top_p,
                                                    device=device
                                                    )

                sample_output = tokenizer.decode(ids=sample_output_tokens[0])

                print(f"Sample output: {sample_output}")

                save_checkpoint(model=model,
                                optimizer=optimizer,
                                iteration=global_step,
                                val_loss=loss_eval,
                                filepath=f"checkpoints/checkpoint_{global_step}.pt"
                                )

            global_step += 1


if __name__ == "__main__":
    main()