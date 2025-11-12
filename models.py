"""Model factory and simple regressors for different architectures."""
from __future__ import annotations

import torch
from torch import nn

# Reuse existing LSTM implementation
from lstm import LSTMRegressor  # noqa: F401


class LinearRegressor(nn.Module):
    """Single linear layer on flattened (T*F) sequence."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LazyLinear(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPRegressor(nn.Module):
    """Multi-layer perceptron on flattened (T*F) sequence."""

    def __init__(self, hidden_size: int, num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten(start_dim=1), nn.LazyLinear(hidden_size), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(
    model_type: str,
    *,
    in_features: int,
    hidden_size: int,
    num_layers: int = 1,
    dropout: float = 0.0,
    bidirectional: bool = False,
    head_hidden_size: int | None = None,
    head_dropout: float = 0.0,
    layer_norm: bool = False,
) -> nn.Module:
    mt = (model_type or "lstm").lower()
    if mt == "lstm":
        return LSTMRegressor(
            in_features=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            head_hidden_size=head_hidden_size,
            head_dropout=head_dropout,
            layer_norm=layer_norm,
        )
    if mt == "linear":
        return LinearRegressor()
    if mt == "mlp":
        return MLPRegressor(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    raise ValueError(f"Unknown model_type '{model_type}'. Use one of: lstm, mlp, linear")

