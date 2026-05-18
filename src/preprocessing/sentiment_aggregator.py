# src/preprocessing/sentiment_aggregator.py

import pandas as pd
from pathlib import Path
from ..utils import ensure_dir, default_logger

def aggregate(inp: Path, outp: Path, log=None):
    # 1. Читаем raw-json
    df = pd.read_json(inp)

    # 2. Если пустой, создаём пустой avg-файл
    if df.empty:
        empty = pd.DataFrame(columns=["date", "avg_sentiment"])
        with ensure_dir(outp).open("w", encoding="utf-8") as f:
            empty.to_json(f, orient="records", indent=4, force_ascii=False)
        log(f"Нет новостей в {inp.name} → создан пустой {outp.name}")
        return

    # 3. Находим колонку с датой и приводим к pd.Timestamp
    #    - если числовые ms-таймстемпы, используем unit="ms"
    if "date" not in df.columns:
        for alt in ("Date","DATE","published","pubDate"):
            if alt in df.columns:
                df = df.rename(columns={alt: "date"})
                break
    if "date" not in df.columns:
        raise KeyError(f"В файле {inp.name} нет столбца 'date'")

    ser = df["date"]
    if pd.api.types.is_integer_dtype(ser.dtype) or pd.api.types.is_float_dtype(ser.dtype):
        dt = pd.to_datetime(ser, unit="ms", errors="coerce")
    else:
        dt = pd.to_datetime(ser, format="%Y-%m-%d", errors="coerce")
    df["date"] = dt

    # 4. Группируем по дате и считаем средний сентимент
    #    При этом dt.date → чистая дата без времени
    out = (
        df.assign(date=df["date"].dt.date)
        .groupby("date")
        .agg({
            "sentiment": [
                "mean",
                "std",
                "min",
                "max",
                "count"
            ]
        })
        .reset_index()
    )

    out.columns = [
        "date",
        "avg_sentiment",
        "std_sentiment",
        "min_sentiment",
        "max_sentiment",
        "news_count"
    ]

    out["std_sentiment"] = (
        out["std_sentiment"]
        .fillna(0)
    )

    # 5. Приводим дату к строке "YYYY-MM-DD"
    out["date"] = out["date"].astype(str)

    # 6. Сохраняем
    with ensure_dir(outp).open("w", encoding="utf-8") as f:
        out.to_json(f, orient="records", indent=4, force_ascii=False)

    log(f"avg_sentiment → {outp.name} ({len(out)} дат)")
