"""
Пакует Investing-новости в регрессионный скаляр сентимента (-1..1).
"""
import json, torch
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path
from ..utils import ensure_dir, default_logger, load_cfg

MODEL_PATH = "cointegrated/rubert-tiny-sentiment-balanced"
print(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

def _score(text: str) -> float:

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)[0]

    negative = probs[0].item()
    neutral = probs[1].item()
    positive = probs[2].item()

    score = positive - negative

    return float(score)

def score(inp_news: Path, out_file: Path, log=None):
    news = json.loads(inp_news.read_text(encoding="utf-8"))
    if not news:
        log(f'{inp_news.name}: пусто → пишем нейтральный []')
        ensure_dir(out_file).write_text('[]', encoding='utf-8')
        return
    for i,n in enumerate(news,1):
        text_for_model = (
            n.get("text")
            or n.get("title")
        )
        n["sentiment"] = _score(text_for_model)
        if i%50==0: log(f"  scored {i}/{len(news)}")
    with ensure_dir(out_file).open("w", encoding="utf-8") as f:
        json.dump(news, f, ensure_ascii=False, indent=4)
    log(f"sentiment -> {out_file.name}")
