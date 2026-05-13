from pathlib import Path
import pandas as pd
import requests
from ..utils import ensure_dir, default_logger, ROOT

def _download_miacr(out: Path, log):
    # Полный URL, как в вашем рабочем скрипте
    url = (
        "https://www.cbr.ru/hd_base/mkr/mkr_base/"
        "?UniDbQuery.Posted=True"
        "&UniDbQuery.From=01.01.2010"
        "&UniDbQuery.To=31.12.2024"
        "&UniDbQuery.st=SF"
        "&UniDbQuery.st=HR"
        "&UniDbQuery.st=MB"
        "&UniDbQuery.ob=OB_MIACR_0"
        "&UniDbQuery.ob=OB_MIACR_IG"
        "&UniDbQuery.ob=OB_MIACR_B"
        "&UniDbQuery.Currency=-1"
        "&UniDbQuery.sk=Dd1_"
        "&UniDbQuery.sk=Dd7"
        "&UniDbQuery.sk=Dd30"
        "&UniDbQuery.sk=Dd90"
        "&UniDbQuery.sk=Dd180"
        "&UniDbQuery.sk=Dd360"
    )
    df = pd.read_html(url, decimal=",")[0]
    df = (
        df.rename(columns={"Дата": "date", "1 день": "miacr_1d"})
          .assign(
              date=lambda x: pd.to_datetime(x.date, dayfirst=True),
              miacr_1d=lambda x: pd.to_numeric(x.miacr_1d, errors="coerce")
          )
          .dropna(subset=["date", "miacr_1d"])
    )
    with ensure_dir(out).open("w", encoding="utf-8") as f:
        df[["date", "miacr_1d"]].to_json(f, orient="records", indent=4, force_ascii=False)
    log("miacr_data.json сохранён")

def _download_rtsi(out: Path, log):
    url = "https://iss.moex.com/iss/engines/stock/markets/index/securities/RTSI/candles.json"
    all_data = []
    start = 0
    step = 100

    session = requests.Session()
    retries = requests.adapters.Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)

    while True:
        try:
            resp = session.get(url, params={
                "interval": 60,
                "from": "2010-01-01",
                "till": "2024-12-31",
                "start": start
            }, timeout=10)

            if resp.status_code != 200:
                log(f"MOEX RTSI ошибка: {resp.status_code}")
                break

            js = resp.json()
            batch = js.get("candles", {}).get("data", [])
            cols  = js.get("candles", {}).get("columns", [])

            if not batch:
                break

            all_data += batch
            start += step

        except requests.exceptions.RequestException as e:
            log(f"RTSI fetch error: {e}")
            break

    if not all_data:
        log("Не удалось получить данные RTSI")
        return

    df = pd.DataFrame(all_data, columns=cols)
    df["begin"] = pd.to_datetime(df["begin"])
    with ensure_dir(out).open("w", encoding="utf-8") as f:
        df.to_json(f, orient="records", indent=4, force_ascii=False)
    log(f"rtsi_data.json сохранён ({len(df)} строк)")


def _download_currency(out: Path, log):
    def _cbr(url, name):
        df = pd.read_xml(url)
        df["Curs"] = df.Value.str.replace(",", ".").astype(float)
        df["Date"] = pd.to_datetime(df.Date, dayfirst=True)
        return df.rename(
            columns={"Curs": f"{name}_rate", "Date": "date"}
        )[[ "date", f"{name}_rate" ]]

    usd = _cbr(
        "https://www.cbr.ru/scripts/XML_dynamic.asp?"
        "date_req1=01/01/2010&date_req2=31/12/2024&VAL_NM_RQ=R01235",
        "usd"
    )
    eur = _cbr(
        "https://www.cbr.ru/scripts/XML_dynamic.asp?"
        "date_req1=01/01/2010&date_req2=31/12/2024&VAL_NM_RQ=R01239",
        "eur"
    )
    merged = usd.merge(eur, on="date")
    with ensure_dir(out).open("w", encoding="utf-8") as f:
        merged.to_json(f, orient="records", indent=4, force_ascii=False)
    log("currency_rates.json сохранён")

def run_all(out_dir: Path = ROOT / "data" / "raw" / "macro", log=None):
    if log is None:
        from .. import utils
        log = utils.default_logger
    out_dir.mkdir(parents=True, exist_ok=True)
    _download_miacr(out_dir / "miacr_data.json", log)
    _download_rtsi(out_dir / "rtsi_data.json", log)
    _download_currency(out_dir / "currency_rates.json", log)
