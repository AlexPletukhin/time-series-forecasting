# --- src/utils.py ---
import logging
from pathlib import Path
import yaml
from datetime import datetime
import sys, importlib
sys.modules.setdefault('utils', importlib.import_module(__name__))


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "config" / "settings.yml").read_text())

def ensure_dir(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

# ------------------------
#  Единый логгер приложения
# ------------------------

ROOT = Path(__file__).resolve().parents[1]

# --------- ОДИН общий логгер на всё приложение -------------------
LOG = logging.getLogger("stock_pipeline")
LOG.setLevel(logging.INFO)
LOG.propagate = False               # чтобы не дублировать в root-логгер
_gui_callback = None                # сюда подкручиваем GUI-функцию

def set_gui_logger(cb):         # ← вызовем из окна
    global _gui_callback
    _gui_callback = cb

def init_logging():
    """Вызываем один раз, пока GUI ещё не создан."""
    if LOG.handlers:
        return                      # уже настроено
    h = logging.StreamHandler()     # обычный stdout
    h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    LOG.addHandler(h)

def default_logger(msg: str):
    stamp = datetime.now().strftime("[%H:%M:%S]")
    line  = f"{stamp} {msg}"
    print(line)                 # старый вариант (stdout)
    if _gui_callback:           # дублируем в окно
        _gui_callback(line)
