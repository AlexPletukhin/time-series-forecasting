# src/training/hyperopt.py

import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from ..utils import default_logger
from .models.tcn import TCNRegression
from .models.transformer import TransformerRegressor
from .models.nlinear import NLinear

# девайс
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# реестр доступных моделей
MODEL_REGISTRY = {
    "TCN": TCNRegression,
    "Transformer": TransformerRegressor,
    "NLinear": NLinear,
}

# пространства гиперпараметров
SEARCH_SPACE = {
    "TCN": lambda trial: {
        "input_dim": 1,
        "num_channels": [
            trial.suggest_int("h", 16, 128) * (2 ** i)
            for i in range(trial.suggest_int("layers", 1, 4))
        ],
        "kernel_size": trial.suggest_int("kernel", 2, 8),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
    },
    "Transformer": lambda trial: {
        "input_dim": 1,
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
        "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
        "num_layers": trial.suggest_int("layers", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
    },
    "NLinear": lambda trial: {
        "input_dim": 1,
        "seq_len": trial.suggest_int("seq_len", 12, 48),
        "pred_len": trial.suggest_int("pred_len", 1, 12),
    },
}

def objective(trial, model_name: str, train_loader: DataLoader, val_loader: DataLoader):
    """
    Общая objective для Optuna.
    Строит модель model_name из MODEL_REGISTRY, парсит SEARCH_SPACE[model_name](trial),
    тренирует несколько эпох (_train_and_eval_one_run) и возвращает валид. loss.
    """
    # 1) собираем гиперпараметры модели
    params = SEARCH_SPACE[model_name](trial)
    # добавляем lr и batch_size в search-space, если нужно
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    params["lr"] = lr

    # 2) создаём модель
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(**params).to(DEVICE)

    # 3) тренируем и валидируем
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    best_val = np.inf
    for epoch in range(1, 50):   # несколько эпох для быстрой оценки
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_loss = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                l = criterion(model(xb), yb).item()
                val_loss += l * xb.size(0)
                count += xb.size(0)
        val_loss /= count

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        best_val = min(best_val, val_loss)

    default_logger(f"[optuna] {model_name} trial best_val={best_val:.6f}")
    return best_val
