"""Johansson et al. 2025, §3.2.1: 2-layer unidirectional LSTM, hidden 64."""

from __future__ import annotations

import torch
from torch import nn

from experiment.config import HIDDEN, HORIZON, LSTM_LAYERS, N_CHANNELS


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_features: int = N_CHANNELS,
        hidden: int = HIDDEN,
        layers: int = LSTM_LAYERS,
        horizon: int = HORIZON,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_features = n_features
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, horizon * n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1]).view(-1, self.horizon, self.n_features)
