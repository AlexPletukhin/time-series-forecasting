# src/training/train_model.py

import optuna
import yaml
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import json

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

from ..utils import default_logger
from .hyperopt import objective, MODEL_REGISTRY
from ..metrics import rmse, mape, mae
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_WEIGHTS = "model.pth"
OPTUNA_STUDY = "study.pkl"


class TimeSeriesDataset(Dataset):

    def __init__(self, df: pd.DataFrame, seq_len: int, horizon: int, use_sentiment: bool = False):

        exclude_cols = [
            "begin",
            "end",
            "date",
            "ticker",
            "symbol"
        ]

        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols
        ]

        if not use_sentiment:
            feature_cols = [
                c for c in feature_cols
                if c not in [
                    "avg_sentiment",
                    "delta_sentiment",
                    "sentiment_ma5"
                ]
            ]

        feature_cols = [
            c for c in feature_cols
            if pd.api.types.is_numeric_dtype(df[c])
        ]

        if "close" in feature_cols:
            feature_cols.remove("close")
            feature_cols.insert(0, "close") 

        print("FEATURE COLUMNS:")
        print(feature_cols)

        features = df[feature_cols].astype(np.float32).values

        scaler = StandardScaler()

        features = scaler.fit_transform(features)

        target = df["close"].astype(np.float32).values.reshape(-1, 1)
        target_scaler = StandardScaler()
        self.target_scaler = target_scaler
        target = target_scaler.fit_transform(target).flatten()

        X, y = [], []

        for i in range(len(df) - seq_len - horizon + 1):

            X.append(features[i:i + seq_len])

            y.append(target[i + seq_len - 1 + horizon])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _train_full(model, train_loader, val_loader, cfg):
    criterion = nn.MSELoss()
    params = cfg["model"]["params"][cfg["model"]["name"]]
    lr = params["lr"]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=cfg["model"]["early_stop"])
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val, patience = np.inf, 0
    for epoch in range(1, cfg["model"]["epochs"] + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                with autocast(enabled=torch.cuda.is_available()):
                    l = criterion(model(xb), yb).item()
                total += l * xb.size(0)
                n += xb.size(0)
        val_loss = total / n

        default_logger(f"[Epoch {epoch}] val_loss={val_loss:.6f}")
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, patience = val_loss, 0
            best_state = model.state_dict()
        else:
            patience += 1
            if patience >= cfg["model"]["early_stop"]:
                default_logger("Внимание, ранняя остановка!")
                model.load_state_dict(best_state)
                break

    return model


def fit(
    dataset: Path,
    model_dir: Path,
    force_retrain: bool = False,
    log=None,
    return_predictions: bool = False,
    horizon=None,
    model_name=None):

    if log is None:                           # «тихий» по умолчанию
        log = lambda *_: None

    model_dir.mkdir(parents=True, exist_ok=True)
    weights_fp = model_dir / MODEL_WEIGHTS
    study_fp   = model_dir / OPTUNA_STUDY

    # ───── 1) Конфиг ───────────────────────────────────────────────
    cfg_path  = Path(__file__).parents[2] / "config" / "settings.yml"
    cfg       = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if model_name is None:
        model_name = cfg["model"]["name"]
    seq_len   = cfg["model"]["seq_len"]
    if horizon is None:
        horizon = cfg["model"]["horizon"]
    log(f"ИМЯ МОДЕЛИ: {model_name}")

    # ───── 2) Данные: train / val / test ───────────────────────────
    df = pd.read_json(dataset)

    if model_name == "ARIMA":

                from statsmodels.tsa.arima.model import ARIMA

                close = df["close"].values

                train_size = int(len(close) * 0.9)

                train_series = close[:train_size]
                test_series = close[train_size:]

                params = cfg["model"]["params"]["ARIMA"]

                model = ARIMA(
                    train_series,
                    order=(
                        params["p"],
                        params["d"],
                        params["q"]
                    )
                ).fit()

                pred = model.forecast(len(test_series))

                if return_predictions:

                    prediction_dates = pd.to_datetime(
                        df["begin"].iloc[train_size:],
                        unit="ms"
                    )

                    return pd.DataFrame({
                        "Date": prediction_dates,
                        "Actual": test_series,
                        "Prediction": pred
                    })

                metric_rmse = rmse(test_series, pred)
                metric_mape = mape(test_series, pred)
                metric_mae = mae(test_series, pred)

                log(
                    f"Back-Test  RMSE={metric_rmse:.2f}  |  MAPE={metric_mape:.2f}%  |  MAE={metric_mae:.2f}%"
                )

                return
                
    ds = TimeSeriesDataset(df, seq_len, horizon)

    n_tot   = len(ds)
    n_train = int(0.8 * n_tot)
    n_val   = int(0.1 * n_tot)
    n_test  = n_tot - n_train - n_val

    # хронологический split, без random_split -----------------------
    from torch.utils.data import Subset

    train_ds = Subset(ds, range(0,               n_train))
    val_ds   = Subset(ds, range(n_train,         n_train + n_val))
    test_ds  = Subset(ds, range(n_train + n_val, n_tot))

    train_loader = DataLoader(train_ds,
                            batch_size=cfg["model"]["batch_size"],
                            shuffle=True)
    val_loader   = DataLoader(val_ds,
                            batch_size=cfg["model"]["batch_size"])

    test_X = test_ds[:][0].to(DEVICE)
    test_y = test_ds[:][1].cpu().numpy().ravel()

    # глобальный индекс первого элемента TEST-набора
    test_start_idx = seq_len - 1 + n_train + n_val

    # ───── 3) Загрузка или подбор гиперпараметров ─────────────────
    if (not force_retrain) and weights_fp.exists():
        log(f"Загрузка уже существующих параметров:  {weights_fp}")
        params = cfg["model"]["params"][model_name]
    else:
        if cfg["model"]["hyperopt"]["enabled"]:
            log("Запуск подбора параметров …")
            study = optuna.create_study(direction="minimize",
                                        pruner=optuna.pruners.MedianPruner())
            study.optimize(lambda t: objective(t, model_name, train_loader, val_loader),
                           n_trials=cfg["model"]["hyperopt"]["n_trials"])
            raw = study.best_params
            log(f"[Optuna] Лучшие параметры → {raw}")

            # перевод результатов Optuna → kwargs модели
            
            if model_name == "TCN":
                layers, h = raw["layers"], raw["h"]
                params = {"input_dim": ds.X.shape[-1],
                          "num_channels": [h * 2 ** i for i in range(layers)],
                          "kernel_size": raw["kernel"],
                          "dropout": raw["dropout"],
                          "lr":       raw["lr"]}
            elif model_name == "Transformer":
                params = {"input_dim": ds.X.shape[-1],
                          "d_model":   raw["d_model"],
                          "nhead":     raw["nhead"],
                          "num_layers":raw["layers"],
                          "dropout":   raw["dropout"],
                          "lr":        raw["lr"]}
            elif model_name == "NLinear":
                params = {"input_dim": ds.X.shape[-1],
                          "seq_len":  seq_len,
                          "pred_len": horizon,
                          "lr":       raw["lr"]}
            elif model_name == "LSTM":
                params = {"input_dim":   ds.X.shape[-1],
                          "hidden_size": raw["hidden_size"],
                          "num_layers":  raw["num_layers"],
                          "dropout":     raw["dropout"],
                          "lr":          raw["lr"]}
            else:
                raise ValueError(f"Unknown model {model_name}")

            # сохраняем лучший конфиг
            cfg["model"]["params"][model_name] = params
            cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
            with study_fp.open("wb") as fh:
                pickle.dump(study, fh)
        else:
            log("Используем лучшие параметры для конфига")

            params = cfg["model"]["params"][model_name]

            if model_name != "ARIMA":

                params["input_dim"] = ds.X.shape[-1]
                params["seq_len"] = seq_len

                if model_name == "NLinear":
                    params["pred_len"] = 1
                else:
                    params["pred_len"] = horizon

                if "lr" in params:
                    params["lr"] = float(params["lr"])

    # ───── 4) Создаём / обучаем модель (если нужно) ────────────────
    ModelCls = MODEL_REGISTRY[model_name]
    model    = ModelCls(**{k: v for k, v in params.items() if k != "lr"}).to(DEVICE)

    if (not force_retrain) and weights_fp.exists():
        model.load_state_dict(torch.load(weights_fp, map_location=DEVICE))
    else:
        model = _train_full(model, train_loader, val_loader, cfg)
        torch.save(model.state_dict(), weights_fp)
        log(f"Параметры сохранены → {weights_fp}")

    model.eval()                               # режим оценки

    # ───── 5) Метрики на test ─────────────────────────────────────
    with torch.no_grad():
        pred = model(test_X).cpu().numpy().ravel()

    pred = ds.target_scaler.inverse_transform(
        pred.reshape(-1, 1)
    ).flatten()

    test_y_real = ds.target_scaler.inverse_transform(
        test_y.reshape(-1, 1)
    ).flatten()

    if return_predictions:

        prediction_dates = pd.to_datetime(
            df["begin"].iloc[test_start_idx:test_start_idx + len(pred)],
            unit="ms"
        )

        return pd.DataFrame({
            "Date": prediction_dates,
            "Actual": test_y_real,
            "Prediction": pred
        })

    metric_rmse = rmse(test_y_real, pred)
    metric_mape = mape(test_y_real, pred)
    metric_mae = mae(test_y_real, pred)
    log(f"Back-Test  RMSE={metric_rmse:.2f}  |  MAPE={metric_mape:.2f}%  |  MAE={metric_mae:.2f}%")

    (model_dir / "metrics.json").write_text(
        json.dumps({"rmse": metric_rmse,
                    "mape": metric_mape,
                    "mae": metric_mae}, indent=4)
    )

    # ───── 6) Реалистичный back-test (только TEST часть) ────────────
    commission = 0.001
    slippage   = 0.0005
    threshold  = 0.005
    capital    = 1_000.0

    trades, equity = [], []
    active = None

    opens = df["open"].values
    dates = pd.to_datetime(df["begin"]).dt.date

    pred_start = test_start_idx            # первый бар, где есть прогноз
    pred_end   = pred_start + len(pred)    # последний (исключая)

    for bar in range(pred_start, len(opens) - horizon):

        # --- выход -----------------------------------------------
        if active and bar == active["exit_idx"]:
            exit_px  = opens[bar] * (1 - slippage)
            proceeds = active["qty"] * exit_px * (1 - commission)
            pnl      = proceeds - active["cash_out"]
            capital += proceeds
            trades[-1].update({
                "exit_date":    str(dates[bar]),
                "exit_price":   round(exit_px, 4),
                "pnl":          round(pnl, 2),
                "capital_after":round(capital, 2)
            })
            active = None

        # --- вход -------------------------------------------------
        if active is None and bar < pred_end - 1:
            forecast   = pred[bar - pred_start]
            entry_px   = opens[bar + 1] * (1 + slippage)
            if forecast > entry_px * (1 + threshold):
                qty = int(capital // (entry_px * (1 + commission)))
                if qty:
                    cash_out = qty * entry_px * (1 + commission)
                    capital -= cash_out
                    exit_idx = bar + horizon
                    trades.append({
                        "entry_date":  str(dates[bar + 1]),
                        "entry_price": round(entry_px, 4),
                        "qty":         qty,
                        "cash_out":    cash_out
                    })
                    active = {"qty": qty, "exit_idx": exit_idx, "cash_out": cash_out}

        # --- mark-to-market --------------------------------------
        pos_val = (active["qty"] * opens[bar] * (1 - commission)) if active else 0
        equity.append(float(capital + pos_val))


    # — сохраняем результаты —
    (model_dir / "equity_curve.json").write_text(json.dumps(equity, indent=4))
    debug_csv = model_dir / "debug_trades.csv"
    pd.DataFrame(trades).to_csv(debug_csv, index=False)

    log(f"Сделки сохранены в {debug_csv}")
    log(f"Кривая доходности сохранена в {model_dir/'equity_curve.json'}")




    # ───── 7) Одношаговый forecast вперёд  ────────────────────────
    with torch.no_grad():
        one_step = model(ds.X[-1:].to(DEVICE)).cpu().item()

    one_step = ds.target_scaler.inverse_transform(one_step.reshape(-1, 1)).item()

    forecast_ts = (pd.to_datetime(df["begin"].iloc[-1], unit="ms")
                   + pd.Timedelta(days=horizon))
    (model_dir / "prediction.json").write_text(
        json.dumps({"predicted": float(one_step),
                    "timestamp": int(forecast_ts.timestamp() * 1000)},
                   indent=4)
    )

    if return_predictions:

        prediction_dates = pd.to_datetime(
            df["begin"].iloc[test_start_idx:test_start_idx + len(pred)],
            unit="ms"
        )

        return pd.DataFrame({
            "Date": prediction_dates,
            "Actual": test_y_real,
            "Prediction": pred
        })

    return model.to("cpu")
