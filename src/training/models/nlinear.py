# src/training/models/nlinear.py

import torch
from torch import nn
from .base import BaseModel

class NLinear(BaseModel):
    """
    Простая линейная модель для прогнозирования одного шага:
    на вход получает x: (batch, 1, seq_len) и выдаёт y: (batch, 1)
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int = 1,  # игнорируется, всегда предсказываем 1 шаг
        **_
    ):
        super().__init__()
        # линейный слой: seq_len → 1
        self.linear = nn.Linear(seq_len, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, seq_len) → (batch, seq_len)
        x = x.squeeze(1)
        # out: (batch, 1)
        out = self.linear(x)
        return out
