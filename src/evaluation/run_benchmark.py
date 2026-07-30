from pathlib import Path
import pandas as pd

from .export_predictions import export_predictions

HORIZONS = [5, 15, 30]

DATASET = Path("data/datasets/SBER.json")
OUTPUT_DIR = Path("results")

OUTPUT_DIR.mkdir(exist_ok=True)

for horizon in HORIZONS:

    print(f"\n========== Horizon = {horizon} ==========\n")

    df = export_predictions(
        dataset=DATASET,
        horizon=horizon,
    )

    out_file = OUTPUT_DIR / f"predictions_h{horizon}.csv"

    df.to_csv(out_file, index=False)

    print(f"Saved -> {out_file}")