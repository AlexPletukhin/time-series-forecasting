# src/metrics.py
import numpy as np

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred, eps=1e-9):
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)
