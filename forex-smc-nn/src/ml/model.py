from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Baseline MLP on flattened feature windows."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
