from pathlib import Path

import pandas as pd
import yaml

from src.pipeline import run_for_ticker

HORIZONS = [5, 15, 30]

MODELS = [
    "ARIMA",
    "LSTM",
    "Transformer",
    "TCN",
    "NLinear"
]

cfg_path = Path("config/settings.yml")

for horizon in HORIZONS:

    cfg = yaml.safe_load(cfg_path.read_text())

    cfg["model"]["horizon"] = horizon
    cfg["training"]["force_retrain"] = False

    cfg_path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True)
    )

    print(f"Horizon = {horizon}")
    
    result = None

    for model in MODELS:

        cfg = yaml.safe_load(cfg_path.read_text())

        cfg["model"]["name"] = model

        cfg_path.write_text(
            yaml.safe_dump(cfg, allow_unicode=True)
        )

        print(f"Running {model}")