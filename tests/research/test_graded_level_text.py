"""切り口 × 水準の日本語の文（src/research/graded/level_text.py）とコースのプルダウン（src/scripts/docs/graded_course_filter.py）のテスト。"""

from __future__ import annotations

import json
from pathlib import Path

from src.research.graded import catalog as CAT
from src.research.graded import level_text as LT
from src.scripts.docs import graded_course_filter as CF

VOCAB = json.loads((Path(__file__).resolve().parents[2] / "data/meta/graded/level_vocab.json").read_text(encoding="utf-8"))["levels"]


def test_every_type_and_level_has_a_sentence():
    """全型に雛形があり、語彙の全水準（全型 × 全水準）で、雛形から文が作れる（型の名前を並べただけの文にならない）。"""
    assert {t.id for t in CAT.TYPES} <= set(LT.TEMPLATES)
    assert {t.id for t in CAT.TYPES} <= set(VOCAB)
    missing, rows = [], []
    for tid, levels in VOCAB.items():
        for lv in levels:
            s, fb = LT.describe(tid, lv)
            rows.append(s)
            if fb or not s.endswith(("馬", "馬（当日）")) or "{" in s:
                missing.append((tid, lv, s))
    assert not missing, missing[:20]
    assert len(rows) == sum(len(v) for v in VOCAB.values()) > 1000


def test_every_level_can_lead_a_combination():
    """全水準を組み合わせの前の位置に置いても、つなぐ形が作れる（「でで」「るで」のような崩れが無い）。買いの組の " | " 区切りの空白も落とす。"""
    bad = []
    for tid, levels in VOCAB.items():
        for lv in levels:
            try:
                s = LT.describe_combo([tid, "A01"], [lv, "1着"])
            except ValueError as e:
                bad.append((tid, lv, str(e)))
                continue
            head = s.split("、かつ")[0]
            if "でで" in s or (head.endswith(("るで", "いで", "すで", "うで")) and not head.endswith("くらいで")) or "「" in s:
                bad.append((tid, lv, s))
    assert not bad, bad[:20]
    assert LT.describe_combo(["I06 ", " K02"], ["件数が少ない ", " ディープ系"]) == "騎手の同じコースの騎乗が少なく、かつ父の小系統が ディープ系 の馬"


def test_vocab_matches_catalog_labels():
    """語彙は catalog の水準のラベルと合っている（catalog のラベルが変わったら語彙を作り直す: python -m src.research.graded.level_text <levels.parquet>）。"""
    import numpy as np
    import pandas as pd
    d = pd.DataFrame({"p_fin": [1, 2, 4, 6, 12, np.nan], "prev_region": ["中央"] * 6})
    got = set(CAT.BY_ID["A01"].fn(d).dropna().astype(str))
    assert got - {CAT.UNKNOWN} <= set(VOCAB["A01"])


def test_examples_and_day_of_mark():
    assert LT.describe("A04", "1着")[0] == "2 走前のレースで 1着だった馬"
    assert LT.describe("E02", "4角 後方20%")[0] == "前走で 4 角を後ろから 2 割以内の位置で回った馬"
    assert LT.describe("D01", "中2-4週")[0] == "前走から中2〜4週の馬"
    assert LT.describe("G06", "480-519kg")[0] == "今回の馬体重が 480〜519kgの馬（当日）"     # 当日の値の型には（当日）
    assert LT.describe("B02", "G1×2-3着")[0] == "前走が G1 のレースで、かつ前走で 2〜3着だった馬"
    assert LT.describe("A01", CAT.UNKNOWN)[0] == "前走の着順が分からない馬"
    assert LT.describe("Z99", "x") == ("Z99が「x」の馬", True)                         # 知らない型は fallback
    assert LT.describe_combo(["A01", "G06"], ["1着", "440-479kg"]) == "前走で 1着で、かつ今回の馬体重が 440〜479kgの馬（当日）"
    assert LT.describe_combo(["G06", "A01"], ["440-479kg", "1着"]) == "今回の馬体重が 440〜479kgで（当日）、かつ前走で 1着だった馬"


def test_course_options_order_counts_and_venue():
    opts = CF.options({"a": "東京芝2400", "b": "東京芝2400", "c": "札幌芝1200", "d": "阪神ダ1800", "e": "阪神芝1600外", "f": "中山芝1200外",
                       "g": "中山ダ1200", "h": "中山芝1200内"})
    assert [o["c"] for o in opts] == ["札幌芝1200", "東京芝2400", "中山芝1200内", "中山芝1200外", "中山ダ1200", "阪神芝1600外", "阪神ダ1800"]
    assert opts[1] == {"v": "東京", "c": "東京芝2400", "n": 2}
    assert CF.split_course("阪神芝1600外") == ("阪神", "芝", 1600, "外") and CF.split_course("?") == ("?", "", 0, "")
    assert "</" not in CF.options_json({"a": "</script>芝1"})
