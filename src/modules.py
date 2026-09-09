import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # TODO 1: Define first linear layer projecting from d_model to d_ff
        self.w_1 = nn.Linear(d_model, d_ff)

        # TODO 2: Define second linear layer projecting from d_ff to d_model
        self.w_2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)

        # TODO 3: Implement forward pass:
        # Pass x through Linear 1 -> ReLU -> Dropout -> Linear 2
        # Hint: F.relu(...) or self.w_1(x).relu()
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)

        return x

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        # TODO 4: Initialize learnable parameters gamma (ones) and beta (zeros)
        # Hint: Use nn.Parameter with torch.ones(features) and torch.zeros(features)
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 5: Calculate mean and variance across the last dimension (dim=-1)
        # Keep dimensions so broadcasting works cleanly (keepdim=True)
        mean = torch.mean(x, -1, keepdim=True)
        var = torch.var(x, -1, keepdim=True, unbiased=False)

        # TODO 6: Normalize x, then scale by gamma and shift by beta
        return self.gamma * (x - mean) / ((var + self.eps) ** 0.5) + self.beta

