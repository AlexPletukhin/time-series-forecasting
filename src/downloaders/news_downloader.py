# ── src/downloaders/news_downloader.py ─────────────────────────────
from __future__ import annotations
from pathlib import Path
import json, time, traceback, sys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from ..utils import ensure_dir, default_logger

RU_MONTHS = {'янв.': '01','фев.': '02','мар.': '03','апр.': '04',
             'мая': '05','июн.': '06','июл.': '07','авг.': '08',
             'сен.': '09','окт.': '10','ноя.': '11','дек.': '12'}

def _normalize_date(txt: str) -> str:
    try:
        d, m, y = txt.replace(' г.', '').split()
        return f'{y}-{RU_MONTHS.get(m.lower(),"01")}-{int(d):02d}'
    except Exception:
        return txt
# ------------------------------------------------------------------
def _safe_driver(opts):
    """
    Пытаемся получить webdriver.  Любая ошибка ➜ возвращаем None.
    """
    try:
        return webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                                options=opts)
    except BaseException as e:
        default_logger(f"[news] ❌ Chrome не запустился: {e}")
        return None
# ------------------------------------------------------------------
def _extract_article_text(article_url: str, log) -> str:
    """
    Загружает полный текст статьи Investing.
    """

    opts = webdriver.ChromeOptions()

    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--blink-settings=imagesEnabled=false")

    opts.page_load_strategy = "eager"

    drv = _safe_driver(opts)

    if drv is None:
        return ""

    try:
        drv.set_page_load_timeout(60)
        drv.get(article_url)

        WebDriverWait(drv, 20).until(
            EC.presence_of_element_located(
                (By.TAG_NAME, "body")
            )
        )

        html = drv.page_source
        soup = BeautifulSoup(html, "html.parser")

        paragraphs = []

        selectors = [
            "div.article_WYSIWYG__O0uhw p",
            "div[class*='article'] p",
            "article p"
        ]

        for selector in selectors:
            nodes = soup.select(selector)

            if len(nodes) > 3:
                for p in nodes:
                    txt = p.get_text(" ", strip=True)

                    if len(txt) > 30:
                        paragraphs.append(txt)

                break

        article_text = "\n".join(paragraphs)

        return article_text.strip()

    except Exception as e:
        log(f"[article-text] Ошибка: {e}")
        return ""

    finally:
        drv.quit()

# ------------------------------------------------------------------
def _get_news_from_page(page_url: str, page: int, log) -> list[dict]:
    url  = page_url if page == 1 else f"{page_url}/{page}"
    opts = webdriver.ChromeOptions()
    opts.add_argument("--start-maximized")
    opts.add_argument("--window-position=0,10000")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")

    drv, html = _safe_driver(opts), ""
    if drv:
        try:
            drv.set_page_load_timeout(60)
            drv.get(url)

            try:
                WebDriverWait(drv, 30).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, 'article[data-test=\"article-item\"]'))
                )
            except Exception:
                log(f"⏳ 30 с без <article> – возможно 404. {url}")
        except BaseException as e:
            log(f"[news] Selenium-ошибка {e}")
        finally:
            html = drv.page_source
            drv.quit()

    try:
        Path("debug_page.html").write_text(html, encoding="utf-8")
    except Exception:
        pass

    if not html:
        return []

    soup, news = BeautifulSoup(html, "html.parser"), []
    for art in soup.select('article[data-test=\"article-item\"]'):
        a = art.select_one('a[data-test=\"article-title-link\"]')
        if not a:
            continue
        link = a["href"]
        if not link.startswith("http"):
            link = "https://ru.investing.com" + link
        t = art.select_one('time[data-test=\"article-publish-date\"]')
        article_title = a.get_text(strip=True)

        log(f"Загружаем текст статьи: {article_title[:80]}")

        article_text = _extract_article_text(link, log)
        if len(article_text) < 100:
            log(f"[warning] Пустая статья: {link}")

        news.append({
            "title": article_title,
            "date": _normalize_date(t.get_text(strip=True)) if t else "N/A",
            "link": link,
            "text": article_text
        })
    return news
# ------------------------------------------------------------------
def _load_json(fp: Path) -> list[dict]:
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            default_logger(f"[news] {fp.name} повреждён: {e}")
    return []

def _is_dup(pool: list[dict], item: dict) -> bool:
    return any((item["title"], item["date"], item["link"]) ==
               (n["title"],   n["date"],   n["link"]) for n in pool)
# ------------------------------------------------------------------
def fetch(page_url: str, out_file: Path, log=None):
    log = log or default_logger
    news = _load_json(out_file)
    log(f"Загружено ранее: {len(news)}")

    page, added = 1, 0
    empty_pages, MAX_EMPTY = 0, 2
    while True:
        log(f"Парсим страницу {page}")
        batch = _get_news_from_page(page_url, page, log)

        if not batch:
            empty_pages += 1
            if empty_pages >= MAX_EMPTY:
                log(f"⛔ {empty_pages} подряд страниц без новостей — стоп.")
                break                     # слишком много пустых
            log(f"⬇️  На странице {page} нет новостей — пробуем {page+1}")
            page += 1
            time.sleep(1)
            continue                      # сразу к следующему витку

        empty_pages = 0

        fresh = [n for n in batch if not _is_dup(news, n)]
        if not fresh:
            log("🔁 Всё на странице уже было – стоп.")
            break

        news.extend(fresh)
        added += len(fresh)
        page  += 1
        time.sleep(1)

    if not news:
        log(f"{out_file.name}: пустой JSON создан – новости отсутствуют")

    with ensure_dir(out_file).open("w", encoding="utf-8") as f:
        json.dump(news, f, ensure_ascii=False, indent=4)

    log(f"{out_file.name}: добавлено {added}, всего {len(news)}")
