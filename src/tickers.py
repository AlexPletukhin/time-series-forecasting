import requests
import yaml
from pathlib import Path
from .utils import default_logger

GITHUB_URL = "https://raw.githubusercontent.com/Rollest/Stonks/main/tickers.yml"

# локальный-файл для переопределений
LOCAL_FILE = Path(__file__).resolve().parents[1] / "config" / "local_tickers.yml"


def merged_mapping() -> dict[str, str]:
    default_logger("⚙️ merged_mapping: старт")

    # 1) remote
    remote: dict[str, str] = {}
    try:
        default_logger("⚙️ загружаем remote YAML")
        resp = requests.get(GITHUB_URL, timeout=5)
        resp.raise_for_status()
        data = yaml.safe_load(resp.text)
        if isinstance(data, dict):
            remote = data
            default_logger(f"[tickers] загружено {len(remote)} из remote")
        else:
            default_logger("[tickers] ⚠️ remote YAML не словарь – игнорирую")
    except Exception as e:
        default_logger(f"[tickers] ❌ remote не загружен: {e}")

    # 2) local
    local: dict[str, str] = {}
    try:
        default_logger("⚙️ загружаем local YAML")
        if LOCAL_FILE.exists():
            data = yaml.safe_load(LOCAL_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                local = data
                default_logger(f"[tickers] загружено {len(local)} локальных значений")
            else:
                default_logger("[tickers] ⚠️ локальный YAML не словарь – игнорирую")
        else:
            default_logger("[tickers] локальный файл не найден – пропускаю")
    except Exception as e:
        default_logger(f"[tickers] ❌ локальный YAML не загружен: {e}")

    # 3) объединяем
    try:
        merged = {**remote, **local}
        default_logger(f"[tickers] итоговая таблица: {len(merged)} тикеров")
    except Exception as e:
        default_logger(f"[tickers] ❌ ошибка объединения remote+local: {e}")
        merged = {}

    return merged




def save_local(ticker: str, url: str) -> None:
    """
    Сохраняет (или обновляет) пару ticker→url в локальном overrides-файле.
    """
    LOCAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOCAL_FILE.exists():
        current = yaml.safe_load(LOCAL_FILE.read_text(encoding="utf-8")) or {}
    else:
        current = {}

    current[ticker.upper()] = url
    LOCAL_FILE.write_text(yaml.safe_dump(current, allow_unicode=True))
    default_logger(f"[tickers] сохранена пара Тикер-URL в локальном override-файле: {ticker.upper()} → {url}")
