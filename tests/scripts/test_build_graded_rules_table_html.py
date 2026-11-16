"""重賞のルール一覧の HTML（build_graded_rules_table_html）と、その材料の詰め方（rules_table.pack）のテスト。"""

from __future__ import annotations

import json
import re

from src.research.graded.rules_table import pack
from src.scripts.docs import build_graded_handicapper_html as OLD
from src.scripts.docs import build_graded_rules_table_html as B

PRE = {"frozen_at": "2026-11-01", "types": {"A04": {"z_disc": 5.0}, "B08": {"z_disc": None}},
       "sets": {"dayof": {"used": ["A04", "B08"]}, "preday": {"used": ["A04"]}}}
PRE_ROI = {"frozen_at": "2026-11-14", "overall_roi": {"disc": {"ret_tan": 0.74, "ret_fuku": 0.73}, "conf": {"ret_tan": 0.71, "ret_fuku": 0.72}},
           "versions": [{"name": "回収率版[テスト]", "primary": True, "patterns": [["B08", "前走 古馬との混合", "凡"]]}]}
RACES = [{"rkey": "皐月賞", "grade": "G1", "course": "中山芝2000", "venue": "中山", "surf": "芝", "dist": 2000, "latest": "2024-04-14", "n_main_ed": 9,
          "all_main_n": 150.0, "all_sub_n": 900.0},
         {"rkey": "中山記念", "grade": "G2", "course": "中山芝1800", "venue": "中山", "surf": "芝", "dist": 1800, "latest": "2024-02-25", "n_main_ed": 9,
          "all_main_n": 120.0, "all_sub_n": 800.0}]


def _row(r, t, lv, dr, k, ly, flags, rec, sub_kind=""):
    return {"r": r, "t": t, "lv": lv, "dir": dr, "k": k, "ly": ly, "sub_kind": sub_kind, "rec": rec, "np": sum(rec), "roi_t": 0.8, "roi_f": 0.9,
            "x": 0.2 if dr == "好" else -0.2, "rm": 1.2, "main": [sum(rec), *rec[:3]], "med": [3, 9], "flags": flags, "roi_kept": bool(flags & 4)}


ROWS = [_row(0, "A04", "1着", "好", "好成績", "主", 3, [5, 3, 2, 10]),
        _row(0, "B08", "前走 古馬との混合", "凡", "凡走", "副", 6, [0, 1, 0, 30], "3勝以上"),
        _row(1, "A04", "2着", "好", "参考", "全重賞", 3, [9, 8, 7, 100])]


def _packed():
    return pack(RACES, ROWS, PRE, PRE_ROI)


def test_pack_is_columnar_with_dictionaries():
    P = _packed()
    assert P["n_rows"] == 3 and [t["id"] for t in P["types"]] == ["A04", "B08"]
    C = P["cols"]
    assert all(len(v) == 3 for v in C.values())
    assert C["ly"] == [0, 1, 2] and C["dir"] == [0, 1, 0] and C["k"] == [0, 1, 2]
    assert C["sub_kind"] == [0, 2, 0] and C["flags"] == [3, 6, 3]
    # z が無い型（B08）は、使う型の中の最小の z を重みにする（run.scored と同じ）
    assert {t["id"]: t["w"] for t in P["types"]} == {"A04": 5.0, "B08": 5.0}
    assert P["levels"][C["lv"][1]] == "前走 古馬との混合"
    assert P["primary"] == "回収率版[テスト]"


def test_build_html_is_self_contained_with_all_columns_and_filters(tmp_path):
    src = tmp_path / "graded_rules.json"
    src.write_text(json.dumps(_packed(), ensure_ascii=False), encoding="utf-8")
    out = B.build(src, tmp_path / "graded_rules_table.html")
    h = out.read_text(encoding="utf-8")
    assert not re.search(r"<script[^>]+src=", h) and "<link" not in h
    head = re.search(r'<tr id="th">(.*?)</tr>', h).group(1)
    assert head.count("<th>") == len(B.HEAD) == 23
    for k in ("重賞", "条件（切り口 × 水準）", "好／凡", "集計の階層", "[1-2-3-着外]", "勝率", "連対率", "複勝率", "単勝の回収率", "複勝の回収率", "件数", "縮めた点 x", "採否", "回収率の判定", "消しに使う"):
        assert f"<th>{k}</th>" in h
    for i in ("cv", "cc", "q", "g", "qt", "dir", "kd", "ly", "ad", "jd", "kl", "mn", "theme"):
        assert f'id="{i}"' in h
    assert "function setupCourse" in h and "行の色" in h and "tr.ad-roi" in h and "tr.ad-kill" in h and "ad-perf" not in h   # コースのプルダウン・行の色の凡例
    assert "prefers-color-scheme:dark" in h and 'name="viewport"' in h
    assert "2016〜2024 年（凍結した値）" in h
    # 埋め込んだ JSON が読み戻せる（</ を逃がしている）
    data = re.search(r'<script type="application/json" id="data">(.*?)</script>', h, re.S).group(1)
    P = json.loads(data.replace("<\\/", "</"))
    assert P["n_rows"] == 3
    assert P["lvtext"] and all(not v.startswith(("A04が", "B08が")) for v in P["lvtext"].values())   # 切り口 × 水準の文（雛形で読めた）
    assert P["course_opts"] and all({"v", "c", "n"} <= set(o) for o in P["course_opts"])
    # 見出しの数と、行の描画（cell）のセルの数・並べ替えのキーの数がそろう
    js = B.JS
    cell = js[js.index("function cell"):js.index("function apply")]
    assert cell.count("<td") == len(B.HEAD)
    keys = js[js.index("const KEY=["):js.index("];", js.index("const KEY=["))]
    assert keys.count("=>") == len(B.HEAD)


def test_handicapper_html_links_to_rules_table(tmp_path):
    src = tmp_path / "graded_report.json"
    src.write_text(json.dumps({"prereg": {"frozen_at": "2026-11-01"}, "races": [], "eval_races": []}), encoding="utf-8")
    h = OLD.build(src, tmp_path / "graded_handicapper.html").read_text(encoding="utf-8")
    assert 'href="graded_rules_table.html"' in h


def _synthetic():
    """2 コース・各コースに重賞 1 つ（9 回）と OP・3 勝クラスの別レースがある、小さな frame と水準の表。"""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(1)
    recs, rid = [], 0
    plan = [("中山芝2000", "中山", "R1", "G1", 7, 9), ("中山芝2000", "中山", None, None, 5, 40), ("中山芝2000", "中山", None, None, 4, 10),
            ("京都ダ1800", "京都", "R2", "G3", 7, 9), ("京都ダ1800", "京都", None, None, 5, 40)]
    for course, venue, rkey, grade, cls, n in plan:
        for j in range(n):
            rid += 1
            fin = rng.permutation(14) + 1
            for h in range(14):
                recs.append({"race_id": f"r{rid}", "horse_id": f"h{rid}_{h}", "dt": pd.Timestamp(2016 + j % 9, 5, 1 + j // 9),
                             "rkey": rkey or f"O{rid}", "course": course, "venue": venue, "surf": course[2], "dist": int(course[3:]),
                             "grade": grade, "cls": cls, "fin": int(fin[h]), "field": 14, "pay_ok": bool(rng.random() > 0.05),
                             "ret_tan": float(fin[h] == 1) * rng.uniform(2, 20), "ret_fuku": float(fin[h] <= 3) * rng.uniform(1, 5)})
    d = pd.DataFrame(recs)
    d["ret_tan_c"], d["ret_fuku_c"] = d["ret_tan"].clip(upper=50), d["ret_fuku"].clip(upper=10)
    # 水準は着順に少し寄せて、点が 0 でない行を作る
    L = pd.DataFrame({"A04": np.where(d["fin"] <= 4, "a", rng.choice(["b", "c"], len(d))),
                      "B08": np.where(rng.random(len(d)) < 0.5, "a", "b")}, index=d.index)
    pre = {"shrink": {"m_g": 3.0, "m_s": 100.0, "m_m": 30.0}, "score": {"lam": 0.5},
           "sets": {"preday": {"used": ["A04"]}, "dayof": {"used": ["A04", "B08"]}}}
    pre_roi = {"versions": [{"name": "回収率版[テスト]", "primary": True, "patterns": [["B08", "b", "凡"], ["A04", "b", "凡"]],
                             "params": {"layer": "主＋副"}, "used": ["A04", "B08"]}]}
    return d, L, pre, pre_roi


def test_rule_rows_match_race_tables_on_synthetic_data():
    """rule_rows（コース × レース名 × OP × 水準の集計を足し戻す）が、export.race_tables（重賞ごとに数え直す）と同じ x・件数・回収率になる。"""
    from src.research.graded.export import race_tables
    from src.research.graded.rules_table import rule_rows

    d, L, pre, pre_roi = _synthetic()
    prim = pre_roi["versions"][0]
    pset = {tuple(p) for p in prim["patterns"]}
    races, rows = rule_rows(d, L, pre, pre_roi)
    rt = race_tables(d, L, pre, ["A04", "B08"], pset, prim)
    got = {(races[r["r"]]["rkey"], r["t"], r["lv"]): r for r in rows}
    n = 0
    for race in rt:
        for p in race["patterns"]:
            if p["x"] == 0:
                continue
            r = got[(race["rkey"], p["t"], p["lv"])]
            assert abs(r["x"] - p["x"]) < 1e-3
            assert r["main"] == [sum(p["m"])] + p["m"][:3]
            assert bool(r["flags"] & 4) == (p["rk"] == "回収率で残った")
            if p["ly"] == "主":
                assert r["ly"] == "主" and r["rec"] == p["m"] and r["np"] == p["rm"]["np"]
                assert abs(r["roi_t"] - p["rm"]["t"][0]) < 2e-3 and abs(r["roi_f"] - p["rm"]["f"][0]) < 2e-3
            n += 1
    assert n >= 4
    assert {r["ly"] for r in rows} >= {"主"}
    assert all(r["x"] != 0 for r in rows)


def test_roi_only_type_has_no_weight_and_judgement_has_three_values():
    """回収率版だけで使う型は重みを付けない（—）。判定は 残った／基準は満たすが全重賞の層で使わない／成績だけ の 3 値。"""
    rows = ROWS + [_row(1, "E10", "6着以下", "凡", "参考", "主", 4, [0, 0, 1, 40])]
    P = pack(RACES, rows, PRE, PRE_ROI)
    assert {t["id"]: t["w"] for t in P["types"]}["E10"] is None
    assert "全重賞の層なので回収率版では使わない" in B.JS and "(C.flags[i]&8)" in B.JS


def test_rule_rows_mark_kill_rows_with_primary_kill_rule():
    """主に見る消しの版の基準（ここでは 主だけ・10 走以上・3着内 0）に当たる行だけに 16 が立つ。"""
    from src.research.graded.rules_table import rule_rows

    d, L, pre, pre_roi = _synthetic()
    # 消しに当たる水準を作る: 中山の重賞 R1 で、着外の馬だけが付く水準 "z"
    m = (d["rkey"] == "R1") & (d["fin"] >= 8)
    L.loc[m, "A04"] = "z"
    pk = {"primary": "消し版[テスト]", "frozen_at": "x", "rules": {"stat": "テスト"},
          "versions": [{"primary": True, "params": {"crit": "3着内0", "nmin": 10, "layer": "主だけ", "hits": 1, "roi7": "入れない", "honmei": "そのまま"}}]}
    races, rows = rule_rows(d, L, pre, pre_roi, pk)
    k = [r for r in rows if r["flags"] & 16]
    assert k and all(r["main"][0] >= 10 and r["main"][1:] == [0, 0, 0] and r["x"] < 0 for r in k)
    assert any(races[r["r"]]["rkey"] == "R1" and r["lv"] == "z" for r in k)
    _, rows0 = rule_rows(d, L, pre, pre_roi)
    assert not any(r["flags"] & 16 for r in rows0)
