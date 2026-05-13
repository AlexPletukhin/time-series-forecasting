# ───────── src/pipeline.py ─────────
import sys
import pathlib
import yaml
from typing import Optional
from src import utils

from .downloaders import macro_downloader, trading_downloader, news_downloader
from .preprocessing import (
    preprocess_trading, sentiment_scoring,
    sentiment_aggregator, feature_engineering
)
from .merging  import macro_merger
from .training import train_model
from .tickers  import merged_mapping, save_local

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG  = yaml.safe_load((ROOT / "config" / "settings.yml").read_text())
log  = utils.default_logger


def run_for_ticker(ticker: str, url: Optional[str] = None) -> None:
    ticker = ticker.upper()

    # 1) макро
    macro_dir = ROOT / "data" / "raw" / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)
    macro_downloader.run_all(out_dir=macro_dir, log=log)

    # 2) дневные свечи
    raw_t = ROOT / "data" / "raw"     / ticker
    int_t = ROOT / "data" / "interim" / ticker
    raw_t.mkdir(parents=True, exist_ok=True)
    int_t.mkdir(parents=True, exist_ok=True)

    hist_json  = raw_t / f"{ticker}_history.json"
    daily_json = int_t / f"{ticker}_daily.json"

    log("Test1")
    try:
        trading_downloader.fetch(
            ticker,
            CFG["start_date"],
            CFG["end_date"],
            hist_json,
            log=log
        )
    except Exception as e:
        log(f"trading_downloader: {e}")
        return
    log("TEst2")

    preprocess_trading.run(hist_json, daily_json, log=log)
    log("TEst3")

    # 3) новости + сентимент
    mapping  = merged_mapping()
    log("TEst4")
    news_url = url or mapping.get(ticker)
    if not news_url:
        log(f"URL новостей для {ticker} не задан. Добавьте в tickers.yml или передайте вторым аргументом.")
        return

    # если пользователь передал новый URL – запишем локально
    if url and url != mapping.get(ticker):
        save_local(ticker, url)

    news_json     = raw_t / f"{ticker}_news.json"
    sent_json     = int_t / f"{ticker}_news_sentiment.json"
    sent_avg_json = int_t / f"{ticker}_news_sentiment_avg_by_date.json"

    news_downloader.fetch(news_url, news_json, log=log)
    sentiment_scoring.score(news_json, sent_json, log=log)
    sentiment_aggregator.aggregate(sent_json, sent_avg_json, log=log)

    # 4) технические фичи + merge
    merged_daily = int_t / f"merged_{ticker}_daily.json"
    feature_engineering.merge(
        daily_json, sent_avg_json, merged_daily, log=log
    )

    # 5) макро + объединение
    final_json = ROOT / "data" / "processed" / ticker / f"final_{ticker}.json"
    final_json.parent.mkdir(parents=True, exist_ok=True)
    macro_merger.merge(
        merged_daily,
        macro_dir / "miacr_data.json",
        macro_dir / "rtsi_data.json",
        macro_dir / "currency_rates.json",
        final_json,
        log=log
    )

    # 6) обучение модели
    model_dir = ROOT / "data" / "models" / ticker
    force     = CFG.get("training", {}).get("force_retrain", False)
    train_model.fit(
        dataset=final_json,
        model_dir=model_dir,
        force_retrain=force,
        log=log
    )

    log(f"=== DONE  {ticker} ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline <TICKER> [NEWS_URL]")
        sys.exit(0)
    _t = sys.argv[1]
    _u = sys.argv[2] if len(sys.argv) > 2 else None
    run_for_ticker(_t, _u)
