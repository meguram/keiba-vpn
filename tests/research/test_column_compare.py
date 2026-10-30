"""めぐコラムの比較グラフの集計（src/research/columns/compare_series.py）。"""

from __future__ import annotations

import json

import pandas as pd

from src.research.columns import compare_series as cs


def test_rates_get_wilson_interval_and_two_sample_p():
    a = pd.Series([1.0] * 30 + [0.0] * 70)
    b = pd.Series([1.0] * 10 + [0.0] * 90)
    ma, na, ca, va = cs._stat(a)
    mb, nb, cb, vb = cs._stat(b)
    assert (ma, na, mb, nb) == (0.3, 100, 0.1, 100)
    assert ca[0] < 0.3 < ca[1] and 0 <= cb[0] < 0.1 < cb[1]
    assert cs._p(va, vb, ma, mb) < 0.001                          # 30% vs 10%（各 100 走）は有意
    assert cs._p(va, va, ma, ma) == 1.0                           # 差が無ければ p=1


def test_empty_side_has_no_value_or_p():
    m, n, ci, v = cs._stat(pd.Series([], dtype=float))
    assert (m, n, ci) == (None, 0, None)
    assert cs._p(v, 0.01, m, 0.2) is None


def test_chart_is_spliced_without_reformatting_the_rest(tmp_path, monkeypatch):
    text = '{\n  "slug": "x",\n  "categories": ["a/b"],\n  "body": [\n    { "type": "text", "text": "前" },\n    {\n      "type": "chart",\n      "title": "古い",\n      "xs": ["1"],\n      "series": [\n        { "name": "条件", "values": [0.5] }\n      ]\n    }\n  ]\n}\n'
    (tmp_path / "x.json").write_text(text, encoding="utf-8")
    monkeypatch.setattr(cs, "COLUMNS_DIR", tmp_path)
    pair = cs.Pair("x", "新しい", ["1"], "条件", "それ以外", "top3", lambda d, x: (d["v"], d["v"]), "横", "縦", replace_title="古い")
    chart = pair.compute(pd.DataFrame({"v": [1.0, 0.0]}))
    assert cs.apply(pair, chart) == "replaced"
    out = (tmp_path / "x.json").read_text(encoding="utf-8")
    assert '"categories": ["a/b"]' in out and '{ "type": "text", "text": "前" }' in out      # 他の部分の書式はそのまま
    doc = json.loads(out)
    assert doc["body"][1]["title"] == "新しい" and doc["body"][1]["compare"]["a"] == "条件"
    assert cs.apply(pair, chart) == "updated"                                              # 2 回目は同じグラフを置き換える
    assert len(json.loads((tmp_path / "x.json").read_text(encoding="utf-8"))["body"]) == 2


def test_merged_chart_removes_the_old_one(tmp_path, monkeypatch):
    text = '{\n  "slug": "y",\n  "body": [\n    { "type": "chart", "title": "A", "xs": ["1"], "series": [] },\n    { "type": "chart", "title": "B", "xs": ["1"], "series": [] },\n    { "type": "text", "text": "後" }\n  ]\n}\n'
    (tmp_path / "y.json").write_text(text, encoding="utf-8")
    monkeypatch.setattr(cs, "COLUMNS_DIR", tmp_path)
    chart = {"type": "chart", "title": "AB", "xs": ["1"], "series": []}
    how = cs.apply(cs.Placement("y", replace_title="A", remove_titles=("B",)), chart)
    assert how.startswith("replaced") and "removed" in how
    body = json.loads((tmp_path / "y.json").read_text(encoding="utf-8"))["body"]
    assert [b.get("title") or b.get("text") for b in body] == ["AB", "後"]
