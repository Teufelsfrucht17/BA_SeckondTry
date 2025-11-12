"""LSTM regressor for momentum-based forecasting."""
from __future__ import annotations

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """LSTM-based regressor with optional bidirectionality and MLP head."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        *,
        bidirectional: bool = False,
        head_hidden_size: int | None = None,
        head_dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        proj_size = hidden_size * (2 if bidirectional else 1)
        self.layer_norm = nn.LayerNorm(proj_size) if layer_norm else None

        head_layers: list[nn.Module] = []
        if head_hidden_size and head_hidden_size > 0:
            head_layers.append(nn.Linear(proj_size, head_hidden_size))
            head_layers.append(nn.ReLU())
            if head_dropout and head_dropout > 0:
                head_layers.append(nn.Dropout(head_dropout))
            head_layers.append(nn.Linear(head_hidden_size, 1))
        else:
            head_layers.append(nn.Linear(proj_size, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        if self.layer_norm is not None:
            last_hidden = self.layer_norm(last_hidden)
        return self.head(last_hidden)
