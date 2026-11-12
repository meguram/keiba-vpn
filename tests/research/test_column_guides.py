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
