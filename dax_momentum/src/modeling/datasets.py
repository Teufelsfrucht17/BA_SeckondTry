"""Dataset abstractions for PyTorch training."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Dataset wrapping numpy arrays of sequences and targets."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("Features and targets must have same first dimension")
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(-1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def build_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader[Tuple[torch.Tensor, torch.Tensor]]:
    dataset = SequenceDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
