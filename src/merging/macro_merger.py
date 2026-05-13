# src/merging/macro_merger.py

import pandas as pd
from datetime import datetime
from pathlib import Path
from ..utils import ensure_dir, default_logger

def timestamp_to_date(ts):
    if isinstance(ts, (int, float)):
        return datetime.utcfromtimestamp(ts / 1000).date()
    elif isinstance(ts, pd.Timestamp):
        return ts.date()
    else:
        return pd.to_datetime(ts).date()

def merge(
    main_file: Path,
    miacr_file: Path,
    rtsi_file: Path,
    currency_file: Path,
    out_file: Path,
    log=None,
):
    # === 1. Основной датасет ===
    main_df = pd.read_json(main_file)
    # приводим begin к datetime и дате
    main_df["begin"] = pd.to_datetime(main_df["begin"], unit="ms", errors="ignore")
    main_df["TRADEDATE"] = main_df["begin"].apply(timestamp_to_date).astype(str)

    # === 2. MIACR ===
    miacr_df = pd.read_json(miacr_file)
    # если нет колонки 'date', пытаемся её найти и переименовать
    if "date" not in miacr_df.columns:
        for cand in ("Date", "DATE"):
            if cand in miacr_df.columns:
                miacr_df = miacr_df.rename(columns={cand: "date"})
                break
    # теперь уверенно создаём TRADEDATE
    miacr_df["date"] = pd.to_datetime(miacr_df["date"], errors="coerce")
    miacr_df["TRADEDATE"] = miacr_df["date"].apply(lambda x: x.date()).astype(str)
    miacr_df = miacr_df.drop(columns=["date"]).drop_duplicates(subset="TRADEDATE")

    # === 3. Валюты ===
    currency_df = pd.read_json(currency_file)
    if "date" not in currency_df.columns:
        for cand in ("Date", "DATE"):
            if cand in currency_df.columns:
                currency_df = currency_df.rename(columns={cand: "date"})
                break
    currency_df["date"] = pd.to_datetime(currency_df["date"], errors="coerce")
    currency_df["TRADEDATE"] = currency_df["date"].apply(lambda x: x.date()).astype(str)
    currency_df = currency_df.drop(columns=["date"]).drop_duplicates(subset="TRADEDATE")

    # === 4. RTSI ===
    rtsi_df = pd.read_json(rtsi_file)
    rtsi_df["begin"] = pd.to_datetime(rtsi_df["begin"], unit="ms", errors="ignore")
    rtsi_df = rtsi_df.rename(columns={"close": "rtsi_close"})[["begin", "rtsi_close"]]
    rtsi_df = rtsi_df.drop_duplicates(subset="begin")

    # === 5. Объединяем всё вместе ===
    df = main_df.merge(miacr_df, on="TRADEDATE", how="left")
    df = df.merge(currency_df, on="TRADEDATE", how="left")
    df = df.merge(rtsi_df, on="begin", how="left")

    # === 6. Финальные штрихи ===
    df.fillna(0, inplace=True)
    before = len(df)
    df.drop_duplicates(inplace=True)
    log(f"🧹 Удалено дубликатов: {before - len(df)}")

    # сохраняем
    with ensure_dir(out_file).open("w", encoding="utf-8") as f:
        df.to_json(f, orient="records", indent=4, force_ascii=False)
    log(f"{out_file.name} сохранён")
