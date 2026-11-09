"""検証の結果から作る通説の記事（src/research/columns/folklore_verified.py・folklore_groups.py）の検査。
CI には検証の結果（/home/jovyan/work/folklore_out）が無いので、定義の整合と、合成データでの組み立てだけを確かめる。"""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from src.research.columns import folklore as F
from src.research.columns import folklore_verified as V
from src.research.columns.folklore_groups import GROUPS
from src.research.columns.folklore_specs import SPECS

REPO = Path(__file__).resolve().parents[2]
COLS = REPO / "frontend/content/columns"


@pytest.mark.parametrize("tier,agree,expect", [
    ("S", True, ("supported", False)), ("A", True, ("supported", False)), ("B", True, ("supported", False)),
    ("S", False, ("not_supported", True)), ("B", False, ("not_supported", True)),
    ("C", True, ("partial", False)), ("C", False, ("not_supported", True)),
    ("無", True, ("not_supported", False)), ("無", False, ("not_supported", False)),
])
def test_tier_verdict(tier, agree, expect):
    assert F.tier_verdict(tier, agree) == expect


def test_groups_are_consistent():
    slugs = [g.slug for g in GROUPS]
    assert len(slugs) == len(set(slugs)), "slug の重複"
    assert not set(slugs) & {s.slug for s in SPECS}, "集計で作る記事と slug が重なっている"
    seen: dict[str, str] = {}
    spec_hids = {s.hid for s in SPECS if getattr(s, "hid", None)}
    for g in GROUPS:
        assert g.main in g.all_hids()
        assert len(g.all_hids()) == len(set(g.all_hids())), f"{g.slug}: 型の重複"
        for h in g.all_hids():
            assert not h.startswith(V.EXCLUDED) and h not in V.EXCLUDED, f"{g.slug}: 対象外の型 {h}"
            assert h not in seen, f"{h} が {seen[h]} と {g.slug} の両方にある"
            assert h not in spec_hids, f"{h} は集計で作る記事に使われている"
            seen[h] = g.slug


def test_group_categories_exist_and_files_written():
    ts = (REPO / "frontend/lib/columns.ts").read_text(encoding="utf-8")
    ids = set(re.findall(r'id: "([a-z0-9-]+)"', ts))
    for g in GROUPS:
        for c in g.categories:
            assert all(p in ids for p in c.split("/")), f"{g.slug}: 未定義のカテゴリ {c}"
        for slug, _ in g.links:
            assert (COLS / f"{slug}.json").exists(), f"{g.slug}: リンク先の記事が無い {slug}"
        doc = json.loads((COLS / f"{g.slug}.json").read_text(encoding="utf-8"))
        assert (COLS / "guides" / f"{g.slug}.json").exists()
        types = [b["type"] for b in doc["body"]]
        assert types.count("chart") <= 1 and types.count("table") == 1 and types.count("stats") == 1, g.slug
        assert sum(len(b["text"]) for b in doc["body"] if b["type"] == "text") <= 600, f"{g.slug}: 本文が長い"


def _synthetic():
    def row(hid, dim, key, n1, y, e, adj, adjc, tier, agree):
        return {"hid": hid, "dim": dim, "key": key, "n1": float(n1), "y3_1": y, "p_mkt_1": e, "adj": adj, "adj_c": adjc,
                "adj_tier": tier, "adj_agree": agree}
    rows = [row("A1", "全体", "全体", 5000, 0.30, 0.25, 0.04, 0.03, "A", True),
            row("A2", "全体", "全体", 3000, 0.20, 0.22, -0.01, None, "無", False)]
    for k, n in (("1勝", 2000), ("2勝", 1500), ("3勝", 800), ("OP", 100)):
        rows.append(row("A1", "クラス", k, n, 0.30, 0.25, 0.04, None, "無", True))
    meta = {"A1": {"hid": "A1", "dir": 1, "how": "条件1", "disc": [2016, 2022], "conf": [2023, 2025], "years": list(range(2016, 2026)), "n_elig": 90000, "n_flag": 7000},
            "A2": {"hid": "A2", "dir": -1, "how": "条件2", "disc": [2016, 2022], "conf": [2023, 2025], "years": list(range(2016, 2026)), "n_elig": 90000, "n_flag": 4000}}
    return pd.DataFrame(rows), meta


def test_build_from_synthetic_results():
    tests, meta = _synthetic()
    g = V.Group(slug="t-x", title="テストは走る？", label="条件1", claim="条件1は走る", categories=["upset/conditions"], main="A1", hids=["A1", "A2"])
    doc, guide, res = V.build(g, tests, meta, "2026-11-09")
    assert doc["verdict"] == "supported" and "人気以上に走る" in doc["title"]
    stats = doc["body"][0]["items"]
    assert stats[0]["value"] == "30.0%" and stats[1]["value"] == "25.0%" and stats[2]["value"] == "+4.0pt"
    table = next(b for b in doc["body"] if b["type"] == "table")
    assert len(table["rows"]) == 2 and table["rows"][1][-1] == "格なし"
    assert table["rows"][0][1] == "走る" and table["rows"][1][1] == "割引" and table["rows"][0][-1] == "格 A・通説どおり"
    chart = next(b for b in doc["body"] if b["type"] == "chart")
    assert chart["xs"] == ["1勝", "2勝", "3勝"], "走数 300 未満のセルは出さず、並びは決めた順"
    assert "compare" not in chart
    assert res["tier"] == "A" and guide["guides"][1]["match"]["table"] == V.TABLE_COLUMNS


def test_reversed_when_tier_disagrees():
    tests, meta = _synthetic()
    tests.loc[(tests.hid == "A1") & (tests.dim == "全体"), "adj_agree"] = False
    g = V.Group(slug="t-y", title="テスト", label="条件1", claim="c", categories=["upset/conditions"], main="A1", hids=["A1"])
    doc, _, res = V.build(g, tests, meta, "2026-11-09")
    assert doc["verdict"] == "not_supported" and res["reversed"] and "むしろ" in doc["title"]


def test_folklore_slugs_cover_both_kinds():
    slugs = V.folklore_slugs()
    assert {s.slug for s in SPECS} <= slugs and {g.slug for g in GROUPS} <= slugs


HEADS = {"supported": {"人気以上に走る", "人気ほど走らない"}, "partial": {"傾向はあるが弱い"},
         "not_supported": {"人気どおりで上積みなし", "むしろ人気ほど走らない", "むしろ人気以上に走る"}}


def test_titles_and_numbers_agree_with_verdicts():
    """記事の見出し・判定・数字・表の最初の行（主の型）が食い違っていないか。"""
    for g in GROUPS:
        doc = json.loads((COLS / f"{g.slug}.json").read_text(encoding="utf-8"))
        assert doc["title"].split("——")[-1] in HEADS[doc["verdict"]], g.slug
        table = next(b for b in doc["body"] if b["type"] == "table")
        assert table["columns"] == V.TABLE_COLUMNS and all(len(r) == len(V.TABLE_COLUMNS) for r in table["rows"]), g.slug
        stats = doc["body"][0]["items"]
        first = table["rows"][0]
        assert stats[0]["value"] == first[3] and stats[1]["value"] == first[4] and stats[2]["value"] == first[5], g.slug
        grade = first[-1]
        if doc["verdict"] == "supported":
            assert grade[:3] in ("格 S", "格 A", "格 B"), g.slug
        if doc["verdict"] == "partial":
            assert grade.startswith("格 C"), g.slug


@pytest.mark.skipif(not (V.VERIFY_DIR / "tests.parquet").exists(), reason="検証の結果（ローカル）が無い")
def test_every_group_hid_has_a_result_row():
    tests, meta = V.load_results()
    have = set(tests.loc[tests["dim"] == "全体", "hid"])
    for g in GROUPS:
        for h in g.all_hids():
            assert h in have and h in meta, f"{g.slug}: {h} が検証の結果に無い"


@pytest.mark.skipif(not (V.VERIFY_DIR / "tests.parquet").exists(), reason="検証の結果（ローカル）が無い")
def test_every_verified_type_is_in_some_article():
    """検証済みで記事にできる型が、集計で作る記事か検証から作る記事のどれかに入っていること（検証側が型を足したら気づく）。"""
    tests, meta = V.load_results()
    have = set(tests.loc[tests["dim"] == "全体", "hid"])
    spec_hids = {s.hid for s in SPECS if getattr(s, "hid", None)}
    used = {h for g in GROUPS for h in g.all_hids()}
    missing = [h for h in meta if h in have and not h.startswith(V.EXCLUDED) and h not in spec_hids and h not in used and not meta[h].get("na")]
    assert not missing, f"どの記事にも入っていない型: {missing}"
