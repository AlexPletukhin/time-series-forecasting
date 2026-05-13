# gui/ui_main.py
import sys, json, gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore    import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QTextEdit, QPushButton,
    QHBoxLayout, QVBoxLayout, QMessageBox, QCompleter
)

# ────────── проектные модули ──────────
from src import utils
from src.pipeline import run_for_ticker            # run_for_ticker(ticker, url)
from src.tickers  import merged_mapping, save_local


# ═════════════════════════════════════════════════════════════════════════════
class MplCanvas(FigureCanvasQTAgg):
    """Два subplot'а: цена (ax_price) + equity (ax_equity)."""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, layout="constrained")
        gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        self.ax_price  = fig.add_subplot(gs[0, 0])
        self.ax_equity = fig.add_subplot(gs[1, 0], sharex=self.ax_price)
        super().__init__(fig)

        self.fig = fig          # для autofmt_xdate()
        self.xs, self.ys = [], []
        self.cursor_art  = []

        # события
        self.mpl_connect("motion_notify_event", self._on_move)
        self.mpl_connect("scroll_event",        self._on_scroll)
        self.mpl_connect("button_press_event",  self._on_press)
        self.mpl_connect("button_release_event",self._on_release)
        self.mpl_connect("motion_notify_event", self._on_drag)
        self.setCursor(Qt.CrossCursor)

        self._is_dragging = False
        self._drag_start  = None

    # ─── helpers ──────────────────────────────────────────────────────────
    def set_price_data(self, xs, ys):
        self.xs = pd.to_datetime(xs).to_list()
        self.ys = ys

    # ─── mouse handlers ──────────────────────────────────────────────────
    def _on_scroll(self, ev: MouseEvent):
        if ev.inaxes is not self.ax_price or ev.xdata is None:
            return
        scale = 1/1.2 if ev.button == "up" else 1.2
        x0, x1 = self.ax_price.get_xlim()
        self.ax_price.set_xlim(ev.xdata - (ev.xdata-x0)*scale,
                               ev.xdata + (x1-ev.xdata)*scale)
        self.draw_idle()

    def _on_move(self, ev: MouseEvent):
        if ev.inaxes is not self.ax_price or not self.xs:
            return
        idx = int(np.argmin(np.abs(mdates.date2num(self.xs) - ev.xdata)))
        cx, cy = self.xs[idx], self.ys[idx]
        for art in self.cursor_art:
            art.remove()
        self.cursor_art = [
            self.ax_price.axvline(cx, ls="--", color="gray"),
            self.ax_price.annotate(f"{cx:%Y-%m-%d}\n{cy:.2f}",
                                   xy=(cx, cy), xytext=(15, 15),
                                   textcoords="offset points",
                                   bbox=dict(boxstyle="round", fc="yellow"),
                                   arrowprops=dict(arrowstyle="->"))
        ]
        self.draw_idle()

    def _on_press(self, ev):  # begin drag
        if ev.inaxes is self.ax_price:
            self._is_dragging, self._drag_start = True, ev.xdata

    def _on_release(self, ev):
        self._is_dragging, self._drag_start = False, None

    def _on_drag(self, ev):
        if self._is_dragging and ev.xdata and self._drag_start:
            dx = self._drag_start - ev.xdata
            x0, x1 = self.ax_price.get_xlim()
            self.ax_price.set_xlim(x0 + dx, x1 + dx)
            self._drag_start = ev.xdata
            self.draw_idle()


# ═════════════════════════════════════════════════════════════════════════════
class PipelineWorker(QThread):
    log  = Signal(str)
    done = Signal()

    def __init__(self, ticker: str, url: str):
        super().__init__()
        self.ticker, self.url = ticker, url

    def run(self):
        try:
            utils.default_logger(f"=== СТАРТ {self.ticker} ===")
            run_for_ticker(self.ticker, self.url)          # ← url теперь обязателен
            utils.default_logger(f"=== ПРОЦЕСС ЗАВЕРШЕН  {self.ticker} ===")
        except Exception:
            import traceback
            utils.default_logger("Ошибка:\n" + traceback.format_exc())
        finally:
            self.done.emit()
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════════
class PipelineUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Pipeline")
        self.resize(1000, 650)

        # ─── 1. загружаем список тикеров ────────────────────────────────
        self.ticker_map = merged_mapping()

        # ─── 2. виджеты ввода ───────────────────────────────────────────
        self.ticker_edit = QLineEdit()
        self.ticker_edit.setPlaceholderText("Тикер (например, MTLR)")
        completer = QCompleter(sorted(self.ticker_map.keys()))
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.ticker_edit.setCompleter(completer)

        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://ru.investing.com/…")

        self.run_btn = QPushButton("Запуск")

        left = QVBoxLayout()
        left.addWidget(QLabel("Введите тикер желаемой акции:"))
        left.addWidget(self.ticker_edit)
        left.addSpacing(8)
        left.addWidget(QLabel("Введите ссылку на Investing.com (необязательно):"))
        left.addWidget(self.url_edit)
        left.addStretch(1)
        left.addWidget(self.run_btn)

        # ─── правый блок: график ───────────────────────────────────────
        self.canvas = MplCanvas(self, width=6, height=4)
        right = QVBoxLayout()
        right.addWidget(QLabel("График"))
        right.addWidget(self.canvas, 1)

        # ─── терминал ──────────────────────────────────────────────────
        self.log_box = QTextEdit(readOnly=True)
        term = QVBoxLayout()
        term.addWidget(QLabel("Лог:"))
        term.addWidget(self.log_box, 1)

        # ─── основной layout ──────────────────────────────────────────
        top = QHBoxLayout()
        top.addLayout(left, 1)
        top.addLayout(right, 2)

        main = QVBoxLayout(self)
        main.addLayout(top, 3)
        main.addLayout(term, 1)

        utils.set_gui_logger(self.append_log)      # redirect логов
        self.run_btn.clicked.connect(self.launch)  # сигнал

    # ──────────────────────────────────────────────────────────────────
    @Slot()
    def launch(self):
        tk  = self.ticker_edit.text().strip().upper()
        url = self.url_edit.text().strip()

        if not tk:
            QMessageBox.warning(self, "Ошибка", "Тикер не введён")
            return

        # ─── обработка unknown / override тикеров ───────────────────
        if tk not in self.ticker_map:
            if not url:
                QMessageBox.warning(
                    self, "Неизвестный тикер",
                    "Тикер отсутствует в общем списке.\n"
                    "Введите URL-адрес страницы новостей."
                )
                return
            # добавляем в локальный файл
            save_local(tk, url)
            self.ticker_map[tk] = url
            # обновляем completer
            comp = QCompleter(sorted(self.tickers.keys()))
            comp.setCaseSensitivity(Qt.CaseInsensitive)
            self.ticker_edit.setCompleter(comp)
            self.append_log(f"📝 Добавлен локальный тикер {tk}")

        else:  # тикер существует
            if not url:                       # пользователь не ввёл URL
                url = self.ticker_map[tk]        # берём из mapping
            elif url != self.ticker_map[tk]:     # пользователь переопределил URL
                save_local(tk, url)
                self.ticker_map[tk] = url
                self.append_log(f"🔧 URL для {tk} переопределён локально")

        # ─── запускаем пайплайн ──────────────────────────────────────
        self.run_btn.setEnabled(False)
        self.log_box.clear()

        self.worker = PipelineWorker(tk, url)
        self.worker.log.connect(self.append_log)
        self.worker.done.connect(self.finished)
        self.worker.start()

    # ──────────────────────────────────────────────────────────────────
    @Slot(str)
    def append_log(self, msg: str):
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )

    @Slot()
    def finished(self):
        self.append_log("🎉 Пайплайн завершён")
        self.run_btn.setEnabled(True)
        self.draw_charts()

    # ─── визуализация ───────────────────────────────────────────────
    def draw_charts(self):
        tk = self.ticker_edit.text().strip().upper()
        base = Path.cwd()
        df_fp   = base / "data" / "processed" / tk / f"final_{tk}.json"
        mod_dir = base / "data" / "models" / tk

        if not df_fp.exists():
            self.append_log(f"Не найден файл {df_fp}")
            return

        df = pd.read_json(df_fp)
        df["begin"] = pd.to_datetime(df["begin"], unit="ms")

        self.canvas.ax_price.clear(); self.canvas.ax_equity.clear()

        # --- price ----------------------------------------------------
        self.canvas.ax_price.plot(df["begin"], df["close"],
                                  color="#1f77b4", label="Цена закрытия")
        self.canvas.ax_price.set_title(f"{tk} Цена закрытия")
        self.canvas.ax_price.legend(loc="upper left")
        self.canvas.set_price_data(df["begin"], df["close"])

        # --- forecast -------------------------------------------------
        pred_fp = mod_dir / "prediction.json"
        if pred_fp.exists():
            p  = json.loads(pred_fp.read_text())
            ts = pd.to_datetime(p["timestamp"], unit="ms")
            val = p["predicted"]
            self.canvas.ax_price.scatter(ts, val, s=80, marker="D",
                                         color="#d62728", edgecolor="k",
                                         label="Прогноз", zorder=5)
            self.canvas.ax_price.plot([df["begin"].iloc[-1], ts],
                                      [df["close"].iloc[-1], val],
                                      ls="--", lw=1, color="#d62728")
            self.canvas.ax_price.annotate(f"{ts:%Y-%m-%d}\n{val:.2f}",
                                          xy=(ts, val), xytext=(10, 10),
                                          textcoords="offset points",
                                          bbox=dict(boxstyle="round,pad=.2",
                                                    fc="#fff59d"),
                                          arrowprops=dict(arrowstyle="->"))
            self.canvas.ax_price.legend(loc="upper right")

        # --- equity ---------------------------------------------------
        eq_fp = mod_dir / "equity_curve.json"
        if eq_fp.exists():
            eq = np.asarray(json.loads(eq_fp.read_text()), dtype=float)
            if len(eq) > 1:
                self.canvas.ax_equity.plot(
                    df["begin"].iloc[-len(eq):], eq,
                    color="tab:green", lw=1.3, label="Доходность"
                )
                self.canvas.ax_equity.set_ylabel("Доходность")
                self.canvas.ax_equity.legend(loc="upper left")

        # --- metrics --------------------------------------------------
        m_fp = mod_dir / "metrics.json"
        if m_fp.exists():
            m = json.loads(m_fp.read_text())
            self.append_log(f"Back-Test  RMSE={m['rmse']:.2f} | "
                            f"MAPE={m['mape']:.2f}%")

        # --- expected annual return -----------------------------------
        eq_fp = mod_dir / "equity_curve.json"
        if eq_fp.exists():
            try:
                equity = json.loads(eq_fp.read_text())
                if len(equity) > 1:
                    total_return = equity[-1] / equity[0] - 1
                    n_days = len(equity)
                    annual_return = (1 + total_return) ** (252 / n_days) - 1
                    self.append_log(f"Ожидаемая годовая доходность стратегии: {annual_return * 100:.2f}%")

                    # — Buy & Hold —
                    # используем цены закрытия из основного датафрейма
                    bh_start = df["close"].iloc[0]
                    bh_end   = df["close"].iloc[-1]
                    bh_total = bh_end / bh_start - 1
                    # приводим к той же длине периода
                    bh_annual = (1 + bh_total) ** (252 / len(df)) - 1
                    self.append_log(f"Buy & Hold годовая доходность: {bh_annual * 100:.2f}%")

                    # — сравнение —
                    diff = (annual_return - bh_annual) * 100
                    sign = "+" if diff >= 0 else ""
                    self.append_log(f"Сверхдоходность стратегии над Buy & Hold: {sign}{diff:.2f}%")

            except Exception as e:
                self.append_log(f"⚠️ Ошибка при расчёте доходности: {e}")


        self.canvas.fig.autofmt_xdate()
        self.canvas.draw_idle()


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui  = PipelineUI()
    ui.show()
    sys.exit(app.exec())
