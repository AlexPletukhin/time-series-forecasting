import requests
import json
import traceback
from pathlib import Path
from ..utils import ensure_dir, default_logger

def fetch(
    ticker: str,
    start_date: str,
    end_date: str,
    out_file: Path,
    log=None
):
    """
    Скачивает ДНЕВНЫЕ свечи MOEX через ISS Candles API.
    Все лог-вызовы «зашиты» в try/except, чтобы сбой в логировании
    не останавливал работу.
    """
    if log is None:
        log = default_logger

    def _safe_log(msg: str):
        try:
            log(msg)
        except Exception:
            # можем дублировать в stdout, но главное — не рвать остальное
            print(f"[log-warning] {msg}")

    base_url = (
        f"https://iss.moex.com/iss/engines/stock/"
        f"markets/shares/securities/{ticker}/candles.json"
    )
    all_data = []
    start = 0
    step  = 100

    _safe_log(f"→ Начинаем загрузку дневных свечей для {ticker} [{start_date}…{end_date}]")

    while True:
        _safe_log(f"  • Запрос {ticker}: from={start_date}, till={end_date}, start={start}, limit={step}")

        # 1) HTTP GET
        try:
            resp = requests.get(
                base_url,
                params={
                    "from":     start_date,
                    "till":     end_date,
                    "interval": 24,
                    "start":    start,
                    "limit":    step
                },
                timeout=15
            )
            resp.raise_for_status()
        except Exception as e:
            _safe_log(f"❌ HTTP ошибка при запросе {base_url}: {e}")
            break

        # 2) JSON-парсинг
        try:
            js = resp.json()
        except Exception as e:
            _safe_log(f"❌ Не удалось распарсить JSON: {e}")
            break

        # 3) Проверка структуры
        candles = js.get("candles")
        if not isinstance(candles, dict):
            _safe_log("⚠️ Ответ не содержит раздела 'candles' или он в неверном формате")
            break

        rows = candles.get("data", [])
        cols = candles.get("columns", [])

        _safe_log(f"    — Получено {len(rows)} строк")

        if not rows:
            _safe_log("ℹ️ Данных больше нет (или их никогда не было)")
            break

        # 4) Добавляем
        try:
            for row in rows:
                all_data.append(dict(zip(cols, row)))
        except Exception as e:
            _safe_log(f"❌ Ошибка сборки записей: {e}\n{traceback.format_exc()}")
            break

        # 5) Завершаем, если всё
        if len(rows) < step:
            _safe_log("ℹ️ Загрузили последний батч, выходим из цикла")
            break

        start += step

    # 6) Сохраняем итог
    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        _safe_log(f"✅ {out_file.name} сохранён — всего {len(all_data)} записей")
    except Exception as e:
        _safe_log(f"❌ Ошибка записи файла {out_file}: {e}\n{traceback.format_exc()}")
