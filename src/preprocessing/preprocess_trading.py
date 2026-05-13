import json
import pandas as pd
from pathlib import Path
from ..utils import ensure_dir, default_logger


def run(inp: Path, outp: Path, log=None):
    if log is None:
        log = default_logger

    try:
        raw = json.loads(inp.read_text(encoding="utf-8"))
        if not raw:
            log(f"{inp.name} пустой → пропуск")
            return

        df = pd.DataFrame(raw)

        # Убедимся, что нужные поля есть
        required = {"open", "high", "low", "close", "volume", "value", "begin"}
        missing = required - set(df.columns.str.lower())
        if missing:
            log(f"{inp.name}: отсутствуют поля {missing} → выход")
            return

        # Преобразуем дату
        df["tradedate"] = pd.to_datetime(df["begin"]).dt.date.astype(str)

        # Фильтрация строк без OHLC
        df.dropna(subset=["open", "high", "low", "close"], how="all", inplace=True)
        if df.empty:
            log(f"{inp.name}: нет валидных OHLC-данных → выход")
            return

        # Агрегация по дате
        df = (df.groupby("tradedate")
                .agg(open=("open", "first"), high=("high", "max"),
                     low=("low", "min"), close=("close", "last"),
                     volume=("volume", "sum"), value=("value", "sum"))
                .reset_index())

        df["value"] = df["value"].round(0).astype(int)

        # Сохраняем
        df.to_json(ensure_dir(outp), orient="records", indent=4, force_ascii=False)
        log(f"{outp.name} готов ({len(df)} дней)")

    except Exception as e:
        log(f"❌ Ошибка при обработке {inp.name}: {e}")
