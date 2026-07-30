import torch
from torch import nn
from .base import BaseModel

class NLinear(BaseModel):

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int = 1,
        **_
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        # linear projection for every feature
        self.linear = nn.Linear(seq_len, pred_len)

        # feature fusion
        self.feature_mixer = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x:
        # [B, T, F]

        x = x.permute(0, 2, 1)

        # [B, F, T]

        out = self.linear(x)

        # [B, F, pred_len]

        out = out.permute(0, 2, 1)

        # [B, pred_len, F]

        out = self.feature_mixer(out)

        # [B, pred_len, 1]

        out = out.squeeze(-1)

        # [B, pred_len]

        return out