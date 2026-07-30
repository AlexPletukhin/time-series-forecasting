import torch
import torch.nn as nn
from .base import BaseModel

class LSTMRegressor(BaseModel):

    def __init__(
        self,
        input_dim,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        **_
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1]

        return self.fc(last_hidden)