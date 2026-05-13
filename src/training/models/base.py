# src/training/models/base.py

from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """
    Базовый класс для всех моделей. 
    Тут мы вызываем nn.Module.__init__(),
    чтобы потом спокойно присваивать submodules.
    """
    def __init__(self, **kwargs):
        super().__init__()   # ← инициализируем Module

    @abstractmethod
    def forward(self, x):
        ...
