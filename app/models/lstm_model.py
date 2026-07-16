# -*- coding: utf-8 -*-
"""M1 — LSTM model for future-return prediction (Phase 3).

Contract: input (batch, 20, F) → output (batch,) predicted 5-day return.
"""
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """LSTM regressor over windowed factor matrices.

    Args:
        input_size: Features per timestep (F).
        hidden_size: LSTM hidden units.
        num_layers: Stacked LSTM layers.
        dropout: Dropout between LSTM layers.
    """

    def __init__(self, input_size: int = 30, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, input_size).

        Returns:
            Tensor of shape (batch,) — predicted future return.
        """
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)
