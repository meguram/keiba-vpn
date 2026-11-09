"""めぐコラム『通説シリーズ』（src/research/columns/folklore*.py）と、既存記事を短くする concise.py の検査。"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.columns import concise
from src.research.columns.folklore import Spec, analyze
from src.research.columns.folklore_frame import _rider_is_jockey
from src.research.columns.folklore_specs import SPECS

REPO = Path(__file__).resolve().parents[2]
COLS = REPO / "frontend/content/columns"


def test_rider_matches_jockey_abbreviation():
    assert _rider_is_jockey("鮫島駿", "鮫島克駿")
    assert _rider_is_jockey("横山武", "横山武史")
    assert _rider_is_jockey("ルメール", "ルメール")
    assert _rider_is_jockey("Ｍデム", "Ｍ．デム")
    assert not _rider_is_jockey("助手", "助手")
    assert not _rider_is_jockey("横山和", "横山武史")
    assert not _rider_is_jockey(None, "武豊")


def _frame(n: int, lift: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    flag = rng.random(n) < 0.3
    exp = np.full(n, 0.2)
    top3 = (rng.random(n) < exp + np.where(flag, lift, 0.0)).astype(float)
    return pd.DataFrame({"flag": flag, "top3": top3, "exp_top3": exp, "ret": 0.0})


def _spec(direction: int = 1) -> Spec:
    return Spec(slug="t", title="t", claim="t", categories=["upset/conditions"], universe=lambda d: pd.Series(True, index=d.index),
                groups=[("条件", lambda d: d["flag"])], direction=direction)


def test_verdict_follows_direction_and_size():
    assert analyze(_spec(), _frame(60_000, 0.03))["verdict"] == "supported"
    assert analyze(_spec(), _frame(60_000, -0.03))["verdict"] == "not_supported"
    assert analyze(_spec(-1), _frame(60_000, -0.03))["verdict"] == "supported"
    assert analyze(_spec(), _frame(5_000, 0.0, seed=1))["verdict"] == "not_supported"


def test_small_sample_is_inconclusive():
    f = _frame(800, 0.0)
    f["flag"] = False
    f.loc[:99, "flag"] = True
    assert analyze(_spec(), f)["verdict"] == "inconclusive"


def test_every_spec_has_article_guide_and_known_category():
    ts = (REPO / "frontend/lib/columns.ts").read_text(encoding="utf-8")
    ids = set(re.findall(r'id: "([a-z0-9-]+)"', ts))
    slugs = [s.slug for s in SPECS]
    assert len(slugs) == len(set(slugs)), "slug の重複"
    for s in SPECS:
        assert (COLS / f"{s.slug}.json").exists(), s.slug
        assert (COLS / "guides" / f"{s.slug}.json").exists(), s.slug
        for c in s.categories:
            assert all(p in ids for p in c.split("/")), f"{s.slug}: 未定義のカテゴリ {c}"
        doc = json.loads((COLS / f"{s.slug}.json").read_text(encoding="utf-8"))
        assert sum(b["type"] == "chart" for b in doc["body"]) <= 2, f"{s.slug}: グラフは 2 枚まで"


def test_short_cuts_at_sentence_end():
    t = "あ" * 100 + "。" + "い" * 100 + "。" + "う" * 100 + "。"
    out = concise._short(t, 250)
    assert out.endswith("。") and len(out) <= 250
    assert concise._short("短い文。") == "短い文。"


def test_existing_articles_are_short():
    for path in COLS.glob("*.json"):
        if path.stem in {s.slug for s in SPECS}:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        texts = [b for b in doc["body"] if b["type"] == "text"]
        assert len(doc["body"]) <= 6, path.stem
        assert sum(len(b["text"]) for b in texts) <= 500, path.stem
