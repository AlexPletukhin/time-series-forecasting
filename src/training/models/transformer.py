import torch.nn as nn
from .base import BaseModel

class TransformerRegressor(BaseModel):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout, **_):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, C_in, L) → (L, B, d_model)
        x = x.permute(2, 0, 1)
        h = self.input_proj(x)             # (L, B, d_model)
        y = self.encoder(h)                # (L, B, d_model)
        return self.out(y[-1])             # (B, 1)
