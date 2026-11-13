"""重賞のルールベース予想家: as-of の列（未来の行を足しても過去の値が変わらない・同じ日の結果を含まない）と、
評価・出力の関数（差の区間・払戻の読み込み・HTML の材料・載せるパターンの絞り込み）。"""

import json

import numpy as np
import pandas as pd

from src.research.graded import features as F


def _runs(rows):
    """rows: (race_id, horse_id, 日付, 着順, グレード)。add_horse_history に要る列をそろえる。"""
    d = pd.DataFrame(rows, columns=["race_id", "horse_id", "dt", "fin", "grade"])
    d["dt"] = pd.to_datetime(d["dt"])
    d["year"] = d["dt"].dt.year
    d["surf"], d["dband"], d["pdiff"], d["mixed"] = "芝", "M", 0.0, False
    d["tsec"] = 90.0 + d["fin"].astype(float)
    d["ok"] = d["fin"].notna()
    d["y3"] = (d["fin"] <= 3).astype(float)
    d["y1"] = (d["fin"] == 1).astype(float)
    d["rkey"] = d["race_id"].str[:1]
    return d


BASE = [("A1", "h1", "2016-01-05", 1, "G3"), ("A1", "h2", "2016-01-05", 2, "G3"), ("A1", "h3", "2016-01-05", 3, "G3"), ("A1", "h4", "2016-01-05", 4, "G3"),
        ("B1", "h1", "2016-03-01", 5, "G2"), ("B1", "h2", "2016-03-01", 1, "G2"), ("B1", "h3", "2016-03-01", 2, "G2"),
        ("A2", "h1", "2017-01-05", 2, "G3"), ("A2", "h2", "2017-01-05", 1, "G3")]
FUTURE = [("C1", "h1", "2017-06-01", 1, "G1"), ("C1", "h2", "2017-06-01", 2, "G1"), ("A3", "h1", "2018-01-05", 1, "G3"), ("A3", "h3", "2018-01-05", 2, "G3")]
HIST_COLS = ["p_margin", "p_race_id", "g_n_prev", "g_top3_prev", "g_win_prev", "g1_n_prev", "rep_n_prev", "rep_top3_prev"]


def _hist(rows):
    return F.add_horse_history(_runs(rows)).set_index(["race_id", "horse_id"]).sort_index()


def test_horse_history_does_not_change_when_future_rows_are_added():
    a = _hist(BASE)[HIST_COLS]
    b = _hist(BASE + FUTURE)[HIST_COLS].loc[a.index]
    pd.testing.assert_frame_equal(a, b)


def test_horse_history_excludes_the_race_itself():
    h = _hist(BASE)
    assert h.loc[("A1", "h1"), "g_n_prev"] == 0 and pd.isna(h.loc[("A1", "h1"), "p_race_id"])
    assert h.loc[("B1", "h1"), "g_top3_prev"] == 1 and h.loc[("B1", "h1"), "g_win_prev"] == 1   # A1 の 1 着だけ
    assert h.loc[("A2", "h2"), "rep_n_prev"] == 1 and h.loc[("A2", "h2"), "rep_top3_prev"] == 1   # 同じレース名 A の前の回
    assert h.loc[("A2", "h1"), "p_race_id"] == "B1"
    assert h.loc[("B1", "h1"), "p_margin"] < 0   # 前走（A1）は勝ち → 2 着とのタイム差をマイナスで


def test_asof_counts_exclude_same_day_and_future():
    d = _runs(BASE + [("A1x", "h5", "2016-01-05", 1, "G3")])   # 同じ日の別レース
    d["jockey_id"] = "j1"
    out = F.asof_counts(d.copy(), ["jockey_id"], d["grade"].isin(["G1", "G2", "G3"]), "jgr")
    same_day = out[out["dt"] == pd.Timestamp("2016-01-05")]
    assert (same_day["jgr_n"] == 0).all()                       # 同じ日の結果は入らない
    b1 = out[out["race_id"] == "B1"]
    assert (b1["jgr_n"] == 5).all() and (b1["jgr_y3"] == 4).all()   # 前の日までの 5 走（A1 の 4 頭と A1x）
    d2 = _runs(BASE + [("A1x", "h5", "2016-01-05", 1, "G3")] + FUTURE)
    d2["jockey_id"] = "j1"
    out2 = F.asof_counts(d2.copy(), ["jockey_id"], d2["grade"].isin(["G1", "G2", "G3"]), "jgr")
    k = ["race_id", "horse_id"]
    m = out.merge(out2, on=k, suffixes=("", "_2"))
    assert len(m) == len(out) and (m["jgr_n"] == m["jgr_n_2"]).all() and (m["jgr_y3"] == m["jgr_y3_2"]).all()


def test_race_level_counts_only_wins_between_the_two_races():
    d = F.add_horse_history(_runs(BASE + FUTURE))
    out = F.add_race_level(d).set_index(["race_id", "horse_id"])
    # h3 の今回 A3（2018-01-05）の前走は B1（2016-03-01）。B1 の相手 h1・h2 のうち、間に勝ったのは h1（C1）と h2（A2）
    assert out.loc[("A3", "h3"), "lvl_mates_won"] == 2
    # h1 の B1 の前走は A1。A1 の相手のうち、A1 の翌日から B1 の前日までに勝った馬はいない（h2 の B1 の勝ちは同じ日）
    assert out.loc[("B1", "h1"), "lvl_mates_won"] == 0


def test_entry_level_counts_all_entries_and_keep_runners_keeps_dnf():
    # master の行（着順の付いた馬と中止の馬。取消の d は master に無い）と、レース結果の表の全頭（取消の d を含む）
    d = pd.DataFrame({"race_id": "R", "horse_id": ["a", "b", "c", "e"], "fin": [1, 2, np.nan, 3], "field": 5,
                      "started": [True, True, True, True], "jw": [57, 55, 58, 56]})
    entries = pd.DataFrame({"race_id": "R", "horse_id": ["a", "b", "c", "d", "e"], "jw": [57, 55, 58, 59, 56]})
    d = F.add_entry_level(d, entries).set_index("horse_id")
    assert (d["field_decl"] == 5).all()
    assert not d["jw_max"].any() and d.loc["c", "jw_rank"] == 0.4      # 取消の馬（59kg）も斤量の順位に入る（出走表の時点）
    # entries に無いレースは、d の行だけで作る
    d2 = F.add_entry_level(pd.DataFrame({"race_id": "S", "horse_id": ["x", "y"], "field": 2, "jw": [57, 55]}), entries).set_index("horse_id")
    assert bool(d2.loc["x", "jw_max"]) and d2.loc["y", "jw_rank"] == 1.0
    # その走の結果（着順・発走したか）を変えても、出走表の時点の列は変わらない
    alt = pd.DataFrame({"race_id": "R", "horse_id": ["a", "b", "c", "e"], "fin": [3, 1, 2, np.nan], "field": 5,
                        "started": [True, True, True, False], "jw": [57, 55, 58, 56]})
    d3 = F.add_entry_level(alt, entries).set_index("horse_id")
    pd.testing.assert_frame_equal(d[["field_decl", "jw_rank", "jw_max"]], d3[["field_decl", "jw_rank", "jw_max"]])
    assert "field_decl" not in alt.columns                              # 入力を書き換えない
    # entries に同じ馬が重複していても、行は増えない
    d4 = F.add_entry_level(alt, pd.concat([entries, entries]))
    assert len(d4) == len(alt)
    k = F.keep_runners(pd.DataFrame({"race_id": "R", "horse_id": list("abcd"), "fin": [1, 2, np.nan, np.nan],
                                     "started": [True, True, True, False]}))
    assert set(k["horse_id"]) == {"a", "b", "c"}                        # 取消（d）は外し、中止（c）は残す
    assert (k["field"] == 3).all()
    assert k.set_index("horse_id").loc["c", "y3"] == 0.0               # 中止は着外


def test_diff_ci_and_payoff(tmp_path, monkeypatch):
    from src.research.graded import run as R
    a = pd.DataFrame({"race_id": list("ABCD"), "win": [1, 0, 1, 0], "place": [1, 1, 1, 0], "top3_in3": [2, 1, 2, 1]})
    b = a.assign(win=[0, 0, 0, 0], place=[1, 0, 0, 0], top3_in3=[1, 1, 1, 1])
    for x, ret in ((a, 300), (b, 0)):
        for bt in ("tansho", "fukusho", "umaren", "wide"):
            x[f"ret_{bt}"], x[f"cost_{bt}"] = ret, 100
    out = R.diff_ci(a, b, n_boot=200)
    assert out["races"] == 4 and np.isclose(out["win"]["est"], 0.5) and np.isclose(out["roi_tansho"]["est"], 3.0)
    assert out["win"]["lo"] <= out["win"]["est"] <= out["win"]["hi"]
    p = tmp_path / "payoff.parquet"
    pd.DataFrame({"race_id": ["R1"] * 5, "bet_type": ["tansho", "fukusho", "umaren", "wide", "sanrenpuku"],
                  "combo": ["3", "3", "5-3", "3-10", "1-3-5"], "payout": [450, 160, 1200, 400, 999], "parsed": [True] * 5}).to_parquet(p)
    monkeypatch.setattr(R, "PAYOFFS", [p])
    pay = R.load_payoff()
    assert pay[("R1", "umaren")] == {"3-5": 1200} and pay[("R1", "wide")] == {"3-10": 400} and ("R1", "sanrenpuku") not in pay
    mk = pd.DataFrame({"race_id": "R1", "hn": [3, 5, 10], "rank": [1, 2, 3], "y1": [1, 0, 0], "y3": [1, 1, 1]})
    rt = R.race_metrics(mk, pay).iloc[0]
    assert rt["ret_umaren"] == 1200 and rt["ret_wide"] == 400
    assert len(R.race_metrics(mk, {k: v for k, v in pay.items() if k[1] != "wide"})) == 0   # 4 券種そろわないレースは除く


def test_export_helpers():
    from src.research.graded import export as E
    assert E._rec(10, 2, 1, 1) == [2, 1, 1, 6]
    assert E._rates(10, 2, 1, 1) == [0.2, 0.3, 0.4] and E._rates(0, 0, 0, 0) == [None, None, None]
    assert E._kind("主", 0.3, 3, 5) == "好成績" and E._kind("主", -0.3, 1, 4) == "凡走" and E._kind("副", 0.5, 9, 9) == "参考"


def test_html_slim_and_build(tmp_path):
    from src.scripts.docs import build_graded_handicapper_html as B
    pats = [{"t": "A01", "lv": f"v{i}", "k": "参考", "x": float(i) * (-1) ** i} for i in range(40)] + [{"t": "A01", "lv": "g", "k": "好成績", "x": 0.1}]
    horse = {"hn": 1, "mk": "◎", "sc": 1.0, "ng": 0, "fin": 1, "jk": "", "v6": None, "n_good": 0, "n_bad": 0,
             "p": [{"t": "A01", "lv": "a", "x": 0.1}, {"t": "A01", "lv": "b", "x": -0.9}] + [{"t": "A01", "lv": f"c{i}", "x": 0.01} for i in range(12)]}
    rep = {"prereg": {"frozen_at": "2026-11-13", "shrink": {"m_g": 3, "m_s": 100, "m_m": 30}, "score": {"lam": 0.5, "weight": "z"},
                      "sets": {"preday": {"used": ["A01"], "adopted": ["A01"]}, "dayof": {"used": ["A01"], "adopted": ["A01"]}}},
           "eval": {"versions": {}, "baselines": {}}, "types": [{"id": "A01", "name": "前走の着順"}], "not_built": [],
           "races": [{"rkey": "X", "patterns": pats}], "eval_races": [{"horses": [horse]}]}
    src = tmp_path / "rep.json"
    src.write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    s = B.slim(json.loads(src.read_text(encoding="utf-8")))
    r = s["races"][0]
    assert r["n_ref_all"] == 40 and r["n_ref_shown"] == B.MAX_REF_PER_RACE
    assert sum(p["k"] == "好成績" for p in r["patterns"]) == 1 and len(r["patterns"]) == 1 + B.MAX_REF_PER_RACE
    hp = s["eval_races"][0]["horses"][0]["p"]
    assert len(hp) == B.MAX_PAT_PER_HORSE and hp[0]["lv"] == "b"          # |点| の大きい順
    out = B.build(src, tmp_path / "o.html")
    html = out.read_text(encoding="utf-8")
    assert "<script src=" not in html and 'id="data"' in html and "</script><script>" in html


def test_load_master_started_and_dnf(tmp_path):
    """FOLKLORE_KEEP_DNF=1 の master: 中止の馬（着順なし・オッズが正）は発走扱いで、着外として過去の成績に数える。オッズは残さない。"""
    p = tmp_path / "master.parquet"
    pd.DataFrame({"race_id": ["R1"] * 3, "horse_id": ["a", "b", "c"], "dt": ["2020-01-05"] * 3, "p_dt": [None] * 3, "fin": [1.0, 2.0, np.nan],
                  "odds": [2.0, 5.0, 30.0], "ok": [True, True, False], "y3": [1.0, 1.0, 1.0], "y1": [1.0, 0.0, 0.0], "surf": "芝",
                  "jockey_id": "j", "trainer_id": "t"}).to_parquet(p)
    d = F.load_master(p).set_index("horse_id")
    assert "odds" not in d.columns and d["started"].all()
    assert bool(d.loc["c", "ok"]) and d.loc["c", "y3"] == 0.0 and d.loc["c", "y1"] == 0.0


def test_result_level_columns_are_forbidden_in_catalog():
    from src.research.graded import catalog as C
    assert {"started", "field", "ok", "odds"} <= C.FORBIDDEN_EXACT
