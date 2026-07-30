from pathlib import Path
import pandas as pd
import yaml

from ..training.train_model import fit

MODELS = [
    "ARIMA",
    "LSTM",
    "Transformer",
    "TCN",
    "NLinear",
]

def export_predictions(dataset: Path, horizon: int):

    result = None

    for model in MODELS:

        print(f"Running {model} (h={horizon})")

        prediction_df = fit(
            dataset=dataset,
            model_dir=(Path("data/models") / f"h{horizon}" / model),
            model_name=model,
            horizon=horizon,
            force_retrain=False,
            return_predictions=True,
        )

        if result is None:

            result = prediction_df[["Date", "Actual"]].copy()

        result[model] = prediction_df["Prediction"].values

    return result