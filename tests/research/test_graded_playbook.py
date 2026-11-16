"""重賞ごとの選び方の早見表（src/research/graded/playbook.py と build_graded_race_playbook_html）のテスト。"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from src.research.graded import cycles as CY
from src.research.graded import playbook as PB
from src.scripts.docs import build_graded_race_playbook_html as H

NAMES = ["score_pre", "score_day", "top3_pre", "top5_pre", "x_A01", "x_E05"]


def _table(seed=0):
    rng = np.random.default_rng(seed)
    rows, n = [], 0
    for rk, grade, years in (("R", "G3", (2019, 2021, 2023, 2025, 2026)), ("S <x>", "G1", (2019, 2021, 2023, 2025, 2026)), ("N", "G2", (2025, 2026))):
        for yr in years:
            field = int(rng.integers(8, 12))
            fin = rng.permutation(field) + 1
            for h in range(field):
                rows.append({"row": n, "race_id": f"{rk}{yr}", "rkey": rk, "grade": grade, "year": yr, "dt": pd.Timestamp(f"{yr}-05-0{1 + (rk == 'R')}"),
                             "venue": "東京" if rk != "S <x>" else "京&都", "course": "東京芝1600" if rk != "S <x>" else '京都"芝', "hn": h + 1, "fin": float(fin[h]), "y1": float(fin[h] == 1), "y3": float(fin[h] <= 3),
                             "pay_ok": not (yr == 2021 or (yr == 2026 and rk == "R")), "ret_tan": float(rng.integers(2, 40)) if fin[h] == 1 else 0.0,
                             "ret_fuku": float(rng.integers(1, 8)) if fin[h] <= 3 else 0.0, "score_pre": float(rng.normal()),
                             # 結果を見せるための値（2026 年は無い扱い → 表で「—」）
                             "horse_name": f"馬<{rk}{h}>" if yr == 2025 else None, "odds_final": 1.5 + h if yr == 2025 else np.nan,
                             "pop_final": float(h + 1) if yr == 2025 else np.nan, "field_size": field + 2 if yr == 2025 else np.nan})
                n += 1
    r = pd.DataFrame(rows)
    r["rank_pre"] = r.groupby("race_id")["score_pre"].rank(ascending=False)
    # x は水準の点なので離散（同点が起きる）
    F = np.column_stack([r["score_pre"], rng.normal(size=len(r)), (r["rank_pre"] <= 3), (r["rank_pre"] <= 5), rng.integers(0, 3, len(r)) * 0.1,
                         rng.integers(0, 2, len(r)) * 0.2])
    return CY.Table(r, F, NAMES)


def _cycle(t):
    """cmd_cycles と同じ手順で、最後の report と規則を作る。"""
    cand = [({"score_pre": 1, "x_A01": 0.1}, "1×score_pre ＋ 0.1×A01"), ({"x_A01": 1}, "1×A01"), ({"x_A01": -1}, "-1×A01"), ({"x_E05": 1, "x_A01": 0.5}, "1×E05 ＋ 0.5×A01"),
            ({"top3_pre": 100, "x_E05": -1}, "top3_preの中で -1×E05"), ({"score_day": 1, "x_E05": 0.3}, "1×score_day ＋ 0.3×E05")]   # どれも型を含む（例の水準を書く）
    bst = CY.Best(t)
    W, lab = CY.matrix(t, cand)
    bst.update(W, lab)
    rep = bst.report()
    terms = {nm: d for d, nm in cand}
    rules = {m: {"honmei_by_rkey": {x["rkey"]: CY.rule_of(x["rule"], terms) for x in rep[m]["honmei_by_rkey"]["per_rkey"]},
                 "kill_by_rkey": {x["rkey"]: CY.rule_of(x["rule"], terms) for x in rep[m]["kill_by_rkey"]["per_rkey"]},
                 "honmei_global": CY.rule_of(rep[m]["honmei_global"]["sel"], terms), "kill_global": CY.rule_of(rep[m]["kill_global"]["sel"], terms)}
             for m in ("A", "B")}
    return rep, rules


def test_cards_reproduce_cycle_report_and_count_dev():
    t = _table()
    rep, rules = _cycle(t)
    res = PB.build_cards(t, rules, rep)
    assert [c["rkey"] for c in res["cards"]] == ["N", "R", "S <x>"]
    assert set(res["global"]) == {"B"}                                             # 全体で 1 つの版も照合を通る（A は廃止）
    assert all("A" not in c and all("A" not in x for x in c["races"]) and not any(k.startswith("A_") for k in c["flags"]) for c in res["cards"])
    assert not any(k.startswith("A_") for k in res["summary"])
    n = res["cards"][0]
    assert n["dev_races"] == 0 and n["B"]["honmei"]["default"] and not n["flags"]["B_honmei"] and not n["flags"]["B_kill"]   # B の既定は数えない
    r = res["cards"][1]
    assert r["base"]["eval"]["pay_races"] == 1 and r["base"]["eval"]["races"] == 2   # 2026 年は払戻なし → 回収率の分母は 1 レース
    for c in res["cards"][1:]:
        for m in ("B",):
            ph = next(x for x in rep[m]["honmei_by_rkey"]["per_rkey"] if x["rkey"] == c["rkey"])
            assert c[m]["honmei"]["eval"]["win"] == pytest.approx(ph["win"]) and c[m]["honmei"]["eval"]["roi_t"] == pytest.approx(ph["roi_t"])
            pk = next(x for x in rep[m]["kill_by_rkey"]["per_rkey"] if x["rkey"] == c["rkey"])
            assert c[m]["kill"]["eval"]["precision"] == pytest.approx(pk["precision"])
            assert c[m]["kill"]["eval"]["killed"] == c[m]["kill"]["k"] * 2           # 8 頭以上なので k 頭ずつ
        # 2018〜2024 年は手で数える（前日版の点が最大の馬）
        dv = t.r[(t.r["rkey"] == c["rkey"]) & (t.r["year"] <= 2024)]
        top = dv.loc[dv.groupby("race_id")["score_pre"].idxmax()]
        assert c["base"]["dev"]["rec"][0] == int((top["fin"] == 1).sum()) and c["base"]["dev"]["races"] == 3
        paid = top[top["pay_ok"]]
        assert c["base"]["dev"]["roi_t"] == pytest.approx(paid["ret_tan"].sum() / len(paid))   # 払戻の解けたレースだけで割る


def test_races_recount_matches_card_stats():
    """回ごとの実際の馬から数え直した [1-2-3-着外]・回収率・消した馬の着が、カードの成績と一致する。"""
    t = _table()
    rep, rules = _cycle(t)
    res = PB.build_cards(t, rules, rep)
    for c in res["cards"]:
        assert len(c["races"]) == c["eval_races"]
        for who, kd, e in [("base", "honmei", c["base"]["eval"])] + [(m, kd, c[m][kd]["eval"]) for m in ("B",) for kd in ("honmei", "kill")]:
            g = PB.recount(c["races"], who, kd)
            assert g["rec"] == e["rec"]
            if kd == "honmei":
                assert g["roi_t"] == pytest.approx(e["roi_t"], abs=1e-3) and g["roi_f"] == pytest.approx(e["roi_f"], abs=1e-3)
            else:
                assert g["killed"] == e["killed"]
        for x in c["races"]:
            rr = t.r[t.r["race_id"] == x["race_id"]]
            assert [w["hn"] for w in x["winners"]] == rr.loc[rr["fin"] == 1, "hn"].tolist()
            assert x["field"] == (len(rr) + 2 if x["date"][:4] == "2025" else len(rr))   # 2025 年は field_size の値、無ければ表の頭数
            # 各馬の値を、元の行（馬番で引く）と独立に突き合わせる
            for h_ in [x["base"]] + [q for m in ("B",) for q in [x[m]["honmei"], *x[m]["kill"]]]:
                o = rr[rr["hn"] == h_["hn"]].iloc[0]
                assert h_["fin"] == o["fin"] and (h_["name"] == o["horse_name"] if x["date"][:4] == "2025" else h_["name"] is None)
                assert (h_["odds"] == o["odds_final"]) if x["date"][:4] == "2025" else h_["odds"] is None
                assert h_["tan"] == (round(o["ret_tan"] * 100) if x["pay_ok"] else None)
            for m in ("B",):
                i = int(np.flatnonzero(t.race["race_id"].to_numpy() == x["race_id"])[0])
                kr = rules[m]["kill_by_rkey"][c["rkey"]]
                want = PB.killed_rows(t, CY._vec(t, kr["terms"]), kr["k"], i)
                assert [h["hn"] for h in x[m]["kill"]] == t.r["hn"].to_numpy()[want].tolist()
            y = x["date"][:4]
            assert (x["base"]["odds"] is None) == (y == "2026") and (x["base"]["name"] is None) == (y == "2026")
            assert (x["base"]["tan"] is None) == (not x["pay_ok"])


def test_recount_kill_with_no_horses():
    assert PB.recount([{"pay_ok": True, "B": {"kill": []}}], "B", "kill") == {"killed": 0, "rec": [0, 0, 0, 0]}


def test_check_races_stops_when_horses_do_not_match():
    t = _table()
    rep, rules = _cycle(t)
    c = PB.build_cards(t, rules, rep)["cards"][1]
    bad = json.loads(json.dumps(c))
    bad["races"][0]["B"]["kill"][0]["fin"] = 1.0 if bad["races"][0]["B"]["kill"][0]["fin"] != 1.0 else 9.0
    with pytest.raises(ValueError):
        PB.check_races(bad)
    bad = json.loads(json.dumps(c))
    x = next(x for x in bad["races"] if x["pay_ok"])
    x["B"]["honmei"]["tan"] += 1000
    with pytest.raises(ValueError):
        PB.check_races(bad)


def test_scales_are_within_race():
    t = _table()
    sc = PB.scales(t)
    t.F[:, t.col("score_day")] = np.repeat(np.arange(len(t.starts), dtype=float), np.diff(np.r_[t.starts, len(t.F)]))   # レースごとに一定
    assert PB.scales(t)["score_day"] == pytest.approx(0.0) and sc["x_A01"] > 0


def test_killed_rows_match_cycles_killed_with_ties():
    t = _table()
    for terms in ({"x_A01": 1.0}, {"x_E05": -1.0, "x_A01": 0.5}, {"top3_pre": 100.0, "x_E05": 1.0}):
        w = CY._vec(t, terms)
        for k in (3, 6):
            nk, nb = CY.killed(t, w[:, None], k)
            for i in range(len(t.starts)):
                rows = PB.killed_rows(t, w, k, i)
                assert len(rows) == nk[i, 0] and t.bad[rows].sum() == nb[i, 0]


def test_cards_stop_when_kill_rule_or_rkey_does_not_match():
    t = _table()
    rep, rules = _cycle(t)
    bad = json.loads(json.dumps(rules))
    for key in bad["B"]["kill_by_rkey"]:
        bad["B"]["kill_by_rkey"][key] = bad["B"]["kill_by_rkey"][key] | {"k": 7 - bad["B"]["kill_by_rkey"][key]["k"] + 2}   # 頭数だけ変える
    with pytest.raises(ValueError):
        PB.build_cards(t, bad, rep)
    rep2 = json.loads(json.dumps(rep))
    rep2["B"]["honmei_by_rkey"]["per_rkey"] = rep2["B"]["honmei_by_rkey"]["per_rkey"][1:]
    with pytest.raises(ValueError):
        PB.build_cards(t, rules, rep2)


def test_cards_stop_when_rules_do_not_match_cycle():
    t = _table()
    rep, rules = _cycle(t)
    bad = json.loads(json.dumps(rules))
    k = next(iter(bad["B"]["honmei_by_rkey"]))
    bad["B"]["honmei_by_rkey"][k] = {"label": "x", "terms": {"x_E05": -3.0, "x_A01": 2.0}, "k": None}
    rep2 = json.loads(json.dumps(rep))
    for x in rep2["B"]["honmei_by_rkey"]["per_rkey"]:
        x["win"], x["roi_t"] = 0.123, 0.456
    with pytest.raises(ValueError):
        PB.build_cards(t, bad, rep2)


def test_describe_direction_and_filter():
    d = PB.describe({"top3_pre": 100.0, "x_A01": -1.0, "x_E05": 0.02, "khit_1": -2.0}, ["上限・40走"], sc={"x_A01": 1.0, "x_E05": 1.0, "khit_1": 1.0})
    assert d["filter"] == "前日版の点の上位 3 頭の中から選ぶ"
    tx = [x["text"] for x in d["terms"]]
    assert tx[0].startswith("消しの規則（上限・40走）に当たった数が少ない") and "「前走の着順」の点（A01）が低いほど上" in tx
    assert d["terms"][-1]["small"] is True and "0.001" in d["tie"]
    s2 = PB.describe({"score_pre": 1.0, "x_A01": -4.0}, sc={"score_pre": 20.0, "x_A01": 0.1})
    assert [x["key"] for x in s2["terms"]] == ["score_pre", "x_A01"] and "0.001" not in s2["tie"]   # 並びは重み × ばらつき
    k = PB.describe({"x_A01": 1.0, "score_pre": -1.0}, kind="kill")
    assert [x["text"] for x in k["terms"]] == ["「前走の着順」の点（A01）が低いほど消す", "前日版の点が高いほど消す"]


def test_flags_exclude_B_default():
    h = {"eval": {"roi_t": 1.5}, "dev": {"roi_t": 0.5}, "default": True}
    k = {"eval": {"top3_rate": 0.0}, "default": False}
    c = {"B": {"honmei": h, "kill": k}, "base": {"eval": {"roi_t": 0.9}}}
    f = PB.flags(c)
    assert not f["B_honmei"] and f["B_kill"] and not f["B_both"] and f["B_default"] and not any(x.startswith("A_") for x in f)
    c2 = {"B": {"honmei": h | {"default": False}, "kill": k}, "base": {"eval": {"roi_t": 0.9}}}
    assert PB.flags(c2)["B_both"]


def test_drop_a_removes_every_a_value():
    P = {"cards": [{"A": {}, "B": {}, "races": [{"A": {}, "B": {}}], "flags": {"A_honmei": True, "B_honmei": False}}],
         "summary": {"A_honmei": 1, "B_honmei": 0, "n": 1}, "global": {"A": {}, "B": {}}}
    Q = PB.drop_a(P)
    assert Q["cards"][0].keys() == {"B", "races", "flags"} and Q["cards"][0]["races"] == [{"B": {}}]
    assert Q["cards"][0]["flags"] == {"B_honmei": False} and Q["summary"] == {"B_honmei": 0, "n": 1} and Q["global"] == {"B": {}}


def test_playbook_html(tmp_path):
    t = _table()
    rep, rules = _cycle(t)
    lv = {(int(x), q): ("水準<b>", "主") for x in t.r["row"] for q in ("A01", "E05")}
    res = PB.build_cards(t, rules, rep, levels=lv)
    res |= {"made_at": "2026-11-15T20:00:00+09:00", "tried": 6}
    src = tmp_path / "playbook.json"
    src.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    h = H.build(src, tmp_path / "p.html").read_text(encoding="utf-8")
    assert not re.search(r"<script[^>]+src=", h) and "prefers-color-scheme:dark" in h and 'name="viewport"' in h
    assert "S &lt;x&gt;" in h and "S <x>" not in h and h.count('<article class="card"') == 3
    assert "水準&lt;b&gt;" in h and "水準<b>" not in h and 'data-venue="京&amp;都"' in h and 'data-course="京都&quot;芝"' in h
    assert "全重賞に同じ決め方を使う版" in h and "払戻" in h
    assert 'href="graded_cycles.html"' in h and 'href="graded_handicapper.html"' in h and "全カード共通の注意" in h
    # 回ごとの実際の馬: 馬名のエスケープ、確定オッズの明記、3 着以内の消し馬を目立たせる
    assert "京&amp;都・" in h and "京&都・" not in h   # 回の見出しの会場
    nm = next(x["B"]["honmei"]["name"] for c in res["cards"] for x in c["races"] if x["B"]["honmei"] and x["B"]["honmei"]["name"])
    assert "確定の単勝オッズ" in h and nm.replace("<", "&lt;").replace(">", "&gt;") in h and nm not in h and "（馬名なし）" in h   # B の ◎ の馬名がエスケープされて出る
    n_hit = sum(1 for c in res["cards"] for x in c["races"] if (ks := {h_["hn"]: h_ for m in ("B",) for h_ in x[m]["kill"]})
                for h_ in ks.values() if h_["fin"] <= 3)
    assert h.count("<tr class='hit'>") == n_hit


def test_cycles_and_handicapper_link_to_playbook(tmp_path):
    from src.scripts.docs import build_graded_cycles_html as CYH
    from src.scripts.docs import build_graded_handicapper_html as OLD
    t = _table()
    rep, _ = _cycle(t)
    L = {"note": "n", "goal": {"win": 0.3, "roi_t": 1.2}, "sub_goal": {"precision": 0.95, "killed_per_race": 3}, "races": 10, "rows": 90, "n_features": 6,
         "cycles": [{"cycle": 1, "at": "2026-11-15T16:10:44+09:00", "hypothesis": "h", "report": rep}], "stopped": {"cycle": 1, "reason": "r"}}
    (tmp_path / "c.json").write_text(json.dumps(L, ensure_ascii=False, default=float), encoding="utf-8")
    assert 'href="graded_race_playbook.html"' in CYH.build(tmp_path / "c.json", tmp_path / "c.html").read_text(encoding="utf-8")
    src = tmp_path / "r.json"
    src.write_text(json.dumps({"prereg": {"frozen_at": "x"}, "races": [], "eval_races": []}), encoding="utf-8")
    assert 'href="graded_race_playbook.html"' in OLD.build(src, tmp_path / "h.html").read_text(encoding="utf-8")


def test_playbook_html_drops_a_from_old_values(tmp_path):
    """A（廃止した版）の入った古い playbook.json からも、A の値・印・絞り込みを出さない。"""
    t = _table()
    rep, rules = _cycle(t)
    res = PB.build_cards(t, rules, rep)
    for c in res["cards"]:   # 古い形: A のカード・回ごとの馬・印・要約を足す
        c["A"] = json.loads(json.dumps(c["B"]))
        for x in c["races"]:
            x["A"] = json.loads(json.dumps(x["B"]))
        c["flags"] |= {"A_honmei": True, "A_kill": True, "A_default": False, "A_dev_honmei": True}
    res["summary"] |= {"A_honmei": 3, "A_kill": 3, "A_dev_honmei": 3, "A_dev_n": 3}
    res["global"]["A"] = res["global"]["B"]
    src = tmp_path / "old.json"
    src.write_text(json.dumps(res | {"made_at": "x", "tried": 6}, ensure_ascii=False), encoding="utf-8")
    h = H.build(src, tmp_path / "p.html").read_text(encoding="utf-8")
    for bad in ("data-ahonmei", "data-akill", 'value="ahonmei"', "A の ◎", "<th>A</th>", "bd a", "A で達成", "◎ は A だけ"):
        assert bad not in h, bad
