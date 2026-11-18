"""めぐコラムの読み方ガイド（frontend/content/columns/guides/<slug>.json）が、記事の本文に当たっているかを確かめる。
記事 JSON は集計コードが書き直すので、表の列名や見出しが変わるとガイドが外れる。そのときここで気づく。"""
import json
from pathlib import Path

import pytest

COLS = Path(__file__).resolve().parents[2] / "frontend/content/columns"
GUIDES = COLS / "guides"
GLOSSARY = json.loads((GUIDES / "glossary.json").read_text(encoding="utf-8"))
SLUGS = sorted(p.stem for p in COLS.glob("*.json"))


def _blocks(body):
    for b in body:
        yield b
        if b["type"] == "accordion":
            yield from _blocks(b["blocks"])


def _hit(m, b):
    t = b["type"]
    if "stats" in m:
        return t == "stats"
    if "treemap" in m:
        return t == "treemap"
    if "heatmap" in m:
        return t == "heatmap"
    if "heading" in m:
        return t == "heading" and b["text"] == m["heading"]
    if "text" in m:
        return t in ("text", "note") and b["text"].startswith(m["text"])
    if "chart" in m:
        return t in ("chart", "jockey_distance_chart") and str(b.get("title", "")).startswith(m["chart"])
    if "table" in m:
        return t == "table" and b["columns"] == m["table"]
    raise AssertionError(f"未知の match: {m}")


def test_every_article_has_a_guide_file():
    assert {p.stem for p in GUIDES.glob("*.json")} - {"glossary"} == set(SLUGS)


@pytest.mark.parametrize("slug", SLUGS)
def test_guide_matches_article(slug):
    g = json.loads((GUIDES / f"{slug}.json").read_text(encoding="utf-8"))
    body = json.loads((COLS / f"{slug}.json").read_text(encoding="utf-8"))["body"]
    blocks = list(_blocks(body))
    for t in g.get("terms", []):
        assert t in GLOSSARY, f"用語集に無い語: {t}"
    for e in g.get("guides", []):
        assert e["text"].strip(), "ガイドの本文が空"
        assert any(_hit(e["match"], b) for b in blocks), f"記事に当たらない match: {e['match']}"


@pytest.mark.parametrize("slug", SLUGS)
def test_guide_numbers_are_in_article(slug):
    """ガイドの答え・読み方の数字は、記事の本文と同じ値にする（集計し直したら直す）。"""
    from src.research.columns.concise import stale_numbers
    g = json.loads((GUIDES / f"{slug}.json").read_text(encoding="utf-8"))
    doc = json.loads((COLS / f"{slug}.json").read_text(encoding="utf-8"))
    assert stale_numbers(g, doc) == [], f"{slug}: 記事に無い数字 {stale_numbers(g, doc)}"
    if "stats" in g:
        stats = next((b for b in doc["body"] if b["type"] == "stats"), None)
        assert stats is not None, "記事に数字の囲みが無いのに、ガイドに stats がある"
        pick = g["stats"]
        assert len(pick) <= 3 and len(set(pick)) == len(pick), "stats は重複なしで 3 つまで"
        assert all(0 <= i < len(stats["items"]) for i in pick), "guides の stats の番号が数字の囲みに無い"


def test_stale_numbers_sees_integers_and_boundaries():
    doc = {"body": [{"type": "text", "text": "119 種牡馬・11.5pt・3.5 倍・1,568 組"}], "summary": "", "title": ""}
    g = {"lead": ["答え: 74頭のうち 1.5pt。3.5倍。1568 組。100% 未満。"], "numbers_ok": ["100%"]}
    from src.research.columns.concise import stale_numbers
    assert stale_numbers(g, doc) == ["1.5pt", "74頭"]
    # 差の向きが逆なら古い値（記事 -1.2pt に対してガイド +1.2pt）。符号なしの「1.2pt 低い」は通す
    doc2 = {"body": [{"type": "text", "text": "差は −1.2pt"}]}
    assert stale_numbers({"lead": ["答え: +1.2pt"]}, doc2) == ["+1.2pt"]
    assert stale_numbers({"lead": ["答え: 1.2pt 低い。-1.2pt。"]}, doc2) == []
