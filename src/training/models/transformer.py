import torch.nn as nn
from .base import BaseModel

class TransformerRegressor(BaseModel):

    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_layers,
        dropout,
        **_
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, T, F]
        h = self.input_proj(x)
        # [B, T, d_model]
        y = self.encoder(h)
        # [B, T, d_model]
        return self.out(y[:, -1, :])