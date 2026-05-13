# src/training/models/tcn.py

from torch import nn
import torch
from .base import BaseModel

class Chomp1d(nn.Module):
    """Обрезает лишние элементы по временной размерности."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x: [B, C, T]
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                           if in_channels != out_channels else None)
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_dim: int, num_channels: list[int], kernel_size: int = 3, dropout: float = 0.2, output_dim: int = 1):
        super().__init__()
        layers = []
        in_channels = input_dim
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i  # usually exponential dilation
            padding = (kernel_size - 1) * dilation
            block = TemporalBlock(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                dropout=dropout
            )
            layers.append(block)
            in_channels = out_channels

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure the input is in the correct format (batch_size, num_channels, seq_len)
        x = x.transpose(1, 2)  # (N, seq_len, 1) -> (N, 1, seq_len)
        out = self.network(x)
        out = out[:, :, -1]  # Use the last timestep's output
        return self.linear(out)


class TCNRegression(BaseModel):
    """
    Регрессор на основе TCN; берёт num_inputs×seq_len → предсказывает 1 точку.
    """
    def __init__(self,
                 input_dim: int,
                 num_channels: list[int],
                 kernel_size: int,
                 dropout: float,
                 **_):
        super().__init__()
        # встроенная TCN
        self.tcn = TCN(
            input_dim=input_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            output_dim=1
        )

    def forward(self, x):
        return self.tcn(x)
