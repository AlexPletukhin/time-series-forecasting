import json, numpy as np, pandas as pd
from pathlib import Path
from ..utils import ensure_dir, default_logger

def _add_technicals(df: pd.DataFrame)->pd.DataFrame:
    df = df.sort_values("begin").reset_index(drop=True)
    df["ma_20"]  = df.close.rolling(20,1).mean()
    df["ma_50"]  = df.close.rolling(50,1).mean()
    df["ma_200"] = df.close.rolling(200,1).mean()
    std20 = df.close.rolling(20,1).std()
    df["upper_bb"]=df.ma_20+2*std20
    df["lower_bb"]=df.ma_20-2*std20
    ema12=df.close.ewm(span=12, adjust=False).mean()
    ema26=df.close.ewm(span=26, adjust=False).mean()
    df["macd"]=ema12-ema26
    df["macd_signal"]=df.macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"]=df.macd-df.macd_signal
    delta=df.close.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    roll_up=gain.rolling(14,1).mean(); roll_down=loss.rolling(14,1).mean()
    rs=roll_up/(roll_down+1e-9); df["rsi_14"]=100-(100/(1+rs))
    df["log_return"]=np.log(df.close/df.close.shift(1))
    for lag in (5,10): df[f"close_lag{lag}"]=df.close.shift(lag)
    return df.bfill().fillna(0)

def merge(candles: Path, sent: Path, out_file: Path, log=None):
    if log is None:
        log = default_logger

    # ─── 1. читаем свечи безопасно ──────────────────────────────────
    if not candles.exists() or candles.stat().st_size == 0:
        log(f"[merge] ⚠  {candles.name} отсутствует / пуст → пропуск")
        return

    try:
        tdf = pd.read_json(candles)
    except ValueError as e:
        log(f"[merge] ⚠  {candles.name}: повреждён JSON ({e}) → пропуск")
        return

    if tdf.empty:
        log(f"[merge] {candles.name}: DataFrame пуст → пропуск")
        return

    # нормализуем имена колонок
    tdf.columns = [c.lower() for c in tdf.columns]

    # ─── 2. приводим даты к единому полю begin/end ────────────────
    if "begin" not in tdf.columns:
        if "tradedate" in tdf.columns:
            tdf["begin"] = pd.to_datetime(tdf["tradedate"])
        else:
            log(f"[merge] в {candles.name} нет 'begin' / 'tradedate'")
            return

    if "end" not in tdf.columns:
        tdf["end"] = tdf["begin"]

    # ─── 3. технические индикаторы ────────────────────────────────
    tdf = _add_technicals(tdf)

    # ─── 4. читаем сентимент (может быть пусто) ───────────────────
    if sent.exists() and sent.stat().st_size:
        try:
            snt = pd.read_json(sent)
            # нормализуем имена
            snt.columns = [c.lower() for c in snt.columns]
        except ValueError:
            log(f"[merge]  {sent.name}: плохой JSON → игнорируем")
            snt = pd.DataFrame()
    else:
        snt = pd.DataFrame()

    # гарантируем наличие нужных колонок
    if "date" in snt.columns:
        snt = snt.rename(columns={"date": "tradedate"})
    if "tradedate" not in snt.columns:
        snt["tradedate"] = []          # будет мерджиться без совпадений
    if "avg_sentiment" not in snt.columns:
        snt["avg_sentiment"] = np.nan  # всё равно заменим на 0

    # ─── 5. объединяем ────────────────────────────────────────────
    tdf["tradedate"] = tdf["begin"].dt.date.astype(str)
    if "tradedate" in snt.columns:
        snt["tradedate"] = pd.to_datetime(snt["tradedate"], errors="coerce").dt.date.astype(str)
    merged = tdf.merge(snt[["tradedate", "avg_sentiment"]], on="tradedate", how="left")

    merged["avg_sentiment"]   = merged["avg_sentiment"].fillna(0.0)
    merged["delta_sentiment"] = merged["avg_sentiment"].diff()
    merged["sentiment_ma5"]   = merged["avg_sentiment"].rolling(5, 1).mean()

    # дополнительные тайм-признаки
    merged["hour"]       = merged["begin"].dt.hour
    merged["dayofweek"]  = merged["begin"].dt.dayofweek
    merged["month"]      = merged["begin"].dt.month

    merged.drop_duplicates(subset=["begin", "close"], inplace=True)

    # ─── 6. сохраняем -------------------------------------------------------
    ensure_dir(out_file).write_text(
        merged.to_json(orient="records", indent=4, force_ascii=False),
        encoding="utf-8"
    )
    log(f"[merge] {out_file.name} сохранён ({len(merged)})")
