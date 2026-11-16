"""めぐ分析の「データで予想」タブ（src/research/graded/picks.py・src/research/race_analysis/graded_picks.py）のテスト。
主にリークの検査: B の規則は 2025・2026 年の結果に依らない、点の元の集計は前日までの as-of、未来の回を足しても選定が変わらない、A は出さない。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research.graded import cycles as CY
from src.research.graded import picks as PK
from src.research.graded import tally as TL
from src.research.race_analysis import graded_picks as GP
from tests.research.test_graded_playbook import NAMES, _cycle, _table
from tests.research.test_graded_roi import _frame

REPO = Path(__file__).resolve().parents[2]


def _prereg(rules, track_from="2026-11-16"):
    return {"rules": rules, "frozen_at": "2026-11-15T16:19:26+09:00", "sha256_rules": "x", "track_from": track_from}


def _history(t=None):
    t = t or _table()
    rep, rules = _cycle(t)
    return PK.build_history(t, _prereg(rules), rep), rep, rules, t


def test_frozen_rules_match_their_hash():
    """凍結した規則（リポジトリの prereg_cycles.json）が、凍結のハッシュと一致する。"""
    from src.research.graded.run import _cycles_hash
    P = json.loads((REPO / "src/research/graded/prereg_cycles.json").read_text(encoding="utf-8"))
    assert _cycles_hash(P) == P["sha256_rules"]
    P["rules"]["B"]["honmei_by_rkey"][next(iter(P["rules"]["B"]["honmei_by_rkey"]))]["terms"] = {"score_pre": 2.0}
    assert _cycles_hash(P) != P["sha256_rules"]


def test_B_rules_do_not_depend_on_2025_2026_results():
    """2025・2026 年の着順・払戻をレースの中で入れ替えても、B の規則（重賞ごと・全体で 1 つ）は変わらない。"""
    t = _table()
    _, rules = _cycle(t)
    r2 = t.r.copy()
    rng = np.random.default_rng(7)
    out_cols = ["fin", "y1", "y3", "ret_tan", "ret_fuku"]
    for rid, g in r2[r2["year"] >= 2025].groupby("race_id"):
        p = rng.permutation(len(g))
        r2.loc[g.index, out_cols] = g[out_cols].to_numpy()[p]
    t2 = CY.Table(r2, t.F.copy(), NAMES)
    assert (t2.r["hn"].to_numpy() == t.r["hn"].to_numpy()).all()       # 並びは同じ（点だけで決まる）
    _, rules2 = _cycle(t2)
    assert rules2["B"] == rules["B"]
    assert rules2["A"] != rules["A"]          # 入れ替えは効いている（A は 2025・2026 年に合わせて選ぶので変わる）


def test_tally_is_asof_and_ignores_future_and_same_day_rows():
    """点の元の集計は、その日より前の回だけ。未来の回・同じ日の別のレースを足しても、前の回の値は変わらない。"""
    d, lv = _frame()
    base = TL.tally_type(TL.base_frame(d), lv, np.ones(len(d), dtype=bool))
    extra = []
    for k, dt in enumerate(["2022-01-01", "2023-01-01", "2024-01-01"]):   # 同じ日の別のレースと未来の回
        for h in range(4):
            extra.append({"race_id": f"F{k}", "dt": pd.Timestamp(dt), "rkey": "X", "course": "C", "grade": "G2", "cls": 6, "fin": h + 1, "field": 4,
                          "ret_tan": 50.0 if h == 0 else 0.0, "ret_fuku": 9.0 if h < 3 else 0.0, "ret_tan_c": 50.0 if h == 0 else 0.0,
                          "ret_fuku_c": 9.0 if h < 3 else 0.0, "pay_ok": True})
    d2 = pd.concat([d, pd.DataFrame(extra)], ignore_index=True)
    lv2 = pd.Series(np.where(d2.index % 4 == 0, "a", "b"), index=d2.index)
    t2 = TL.tally_type(TL.base_frame(d2), lv2, np.ones(len(d2), dtype=bool))
    pd.testing.assert_frame_equal(t2.loc[base.index, base.columns], base, check_dtype=False)


def test_history_matches_cycle_report_and_has_no_A():
    h, rep, rules, t = _history()
    assert set(h["rkeys"]) == set(rules["B"]["honmei_by_rkey"])
    for key, v in h["rkeys"].items():
        ev = [x for x in v["races"] if x["date"][:4] >= "2025"]
        ph = next(x for x in rep["B"]["honmei_by_rkey"]["per_rkey"] if x["rkey"] == key)
        s = PK.stats(ev, "B", "honmei")
        assert s["win"] == pytest.approx(ph["win"])
        assert (s["roi_t"] is None and ph["roi_t"] is None) or s["roi_t"] == pytest.approx(ph["roi_t"], abs=1e-3)
        for x in v["races"]:
            assert set(x) >= {"base", "B", "winners"} and "A" not in x
    s = json.dumps(h, ensure_ascii=False)
    assert '"A"' not in s and "honmei_global" not in s     # A の規則・値は持たない


def test_future_races_do_not_change_earlier_picks():
    """ある日より後の回を表から除いても、それより前の回の ◎・消しは同じ（各回の選定はその回の行の点だけで決まる。
    点そのものに未来が入らないことは、test_tally_is_asof_and_ignores_future_and_same_day_rows と features の as-of の検査で見る）。"""
    h, _, rules, t = _history()
    cut = "2024-12-31"
    keep = t.r["dt"] <= pd.Timestamp(cut)
    t_cut = CY.Table(t.r[keep].reset_index(drop=True), t.F[keep.to_numpy()], NAMES)
    rep_full = _cycle(t)[0]
    h_cut = PK.build_history(t_cut, _prereg(rules), _dev_only_report(rep_full, rules))
    for key, v in h_cut["rkeys"].items():
        full = {x["race_id"]: x for x in h["rkeys"][key]["races"]}
        for x in v["races"]:
            y = full[x["race_id"]]
            assert x["B"]["honmei"]["hn"] == y["B"]["honmei"]["hn"] and x["base"]["hn"] == y["base"]["hn"]
            assert [q["hn"] for q in x["B"]["kill"]] == [q["hn"] for q in y["B"]["kill"]]


def _dev_only_report(rep, rules):
    """2025 年以降の回が無い表に当てるための、照合の無い記録（per_rkey の値を None にして照合を飛ばす）。"""
    r = json.loads(json.dumps(rep))
    for kind in ("honmei_by_rkey", "kill_by_rkey"):
        for x in r["B"][kind]["per_rkey"]:
            x.update({"win": None, "roi_t": None, "precision": None})
    return r


def test_stats_definitions():
    hs = lambda fin, tan, fuku: {"hn": 1, "name": None, "odds": None, "pop": None, "fin": fin, "tan": tan, "fuku": fuku}   # noqa: E731
    races = [{"pay_ok": True, "base": hs(1, 380, 180), "B": {"honmei": hs(1, 380, 180), "kill": [hs(4, 0, 0), hs(2, 0, 150)]}},
             {"pay_ok": False, "base": hs(5, None, None), "B": {"honmei": hs(2, None, None), "kill": []}}]
    s = PK.stats(races, "B", "honmei")
    assert s["rec"] == [1, 1, 0, 0] and s["win"] == 0.5 and s["place"] == 1.0 and s["roi_t"] == pytest.approx(3.8) and s["pay_races"] == 1
    k = PK.stats(races, "B", "kill")
    assert k == {"races": 2, "killed": 2, "rec": [0, 1, 0, 1], "top3_rate": 0.5}
    assert PK.period_of("2024-05-01", "2026-11-16") == "dev" and PK.period_of("2026-11-15", "2026-11-16") == "eval"
    assert PK.period_of("2026-11-22", "2026-11-16") == "forward"
    assert PK.needs_dayof({"x_G06": 1.0}) and PK.needs_dayof({"score_day": 1.0}) and not PK.needs_dayof({"x_A01": 1.0, "score_pre": 1})


def _write_history(tmp_path, monkeypatch, h):
    d = tmp_path / "gp"
    d.mkdir()
    (d / "history.json").write_text(json.dumps(h, ensure_ascii=False), encoding="utf-8")
    pr = tmp_path / "prereg_cycles.json"
    pr.write_text(json.dumps({"sha256_rules": h["sha256_rules"], "frozen_at": h["frozen_at"]}), encoding="utf-8")
    monkeypatch.setattr(GP, "PREREG_CYCLES", pr)
    monkeypatch.setenv("GRADED_PICKS_DIR", str(d))
    monkeypatch.setattr("src.research.graded.race_keys.canonical_key", lambda name, month=None: name)


def test_figure_uses_only_earlier_races(tmp_path, monkeypatch):
    h, *_ = _history()
    _write_history(tmp_path, monkeypatch, h)
    v = h["rkeys"]["R"]
    last = max(v["races"], key=lambda x: x["date"])
    f = GP.figure_for(GP.load_history(), "R", last["race_id"], last["date"])
    assert f["this"]["status"] == "result" and f["this"]["race"]["race_id"] == last["race_id"]
    assert all(x["date"] < last["date"] for x in f["past"]) and len(f["past"]) == len(v["races"]) - 1
    # 成績は、このレースより前の回だけで数える
    ev = [x for x in f["past"] if x["date"][:4] >= "2025"]
    assert f["stats"]["eval"]["B_honmei"] == PK.stats(ev, "B", "honmei")
    # まだ記録に無い先の回は「出馬表の確定後」、当日の型を使う規則なら「当日」
    f2 = GP.figure_for(GP.load_history(), "R", "R2027", "2027-05-01")
    assert f2["this"] == {"status": "pending", "race": None, "period": "forward", "picked": None} and len(f2["past"]) == len(v["races"])
    h2 = json.loads(json.dumps(h))
    h2["rkeys"]["R"]["honmei"]["dayof"] = True
    assert GP.figure_for(h2, "R", "R2027", "2027-05-01")["this"]["status"] == "dayof"
    assert GP.figure_for(h, "N", "N2027", "2027-05-01")["this"]["status"] == "default"   # 2018〜2024 年に回が無い重賞（B は既定）


def test_patch_saved_adds_tab_only_to_graded_races(tmp_path, monkeypatch):
    h, *_ = _history()
    _write_history(tmp_path, monkeypatch, h)
    ra = tmp_path / "ra"
    ra.mkdir()
    monkeypatch.setenv("KEIBA_RACE_ANALYSIS_DIR", str(ra))
    other = {"id": "jockey_course", "label": "騎手", "status": "ok"}
    for rid, name, date in (("R2026", "R", "2026-05-02"), ("Z1", "ふつうのレース", "2026-05-02")):
        (ra / f"{rid}.json").write_text(json.dumps({"race": {"race_id": rid, "race_name": name, "date": date}, "entries": [], "status": "ok",
                                                    "modules": [other], "fingerprint": "fp"}, ensure_ascii=False), encoding="utf-8")
    assert GP.patch_saved() == {"patched": 1, "skipped": 1, "error": 0}
    p = json.loads((ra / "R2026.json").read_text(encoding="utf-8"))
    assert [m["id"] for m in p["modules"]] == ["jockey_course", "graded_picks"] and p["fingerprint"] == "fp"   # ほかのタブ・指紋はそのまま
    m = p["modules"][1]
    assert m["status"] == "ok" and m["view"] == "graded_picks" and m["figure"]["this"]["status"] == "result"
    assert any("2024 年までのデータだけで決め" in n for n in m["notes"]) and any("まっさらな検証ではありません" in n for n in m["notes"])
    assert GP.patch_saved() == {"patched": 1, "skipped": 1, "error": 0}             # 2 回目も同じ（入れ替え）
    assert [m["id"] for m in json.loads((ra / "R2026.json").read_text(encoding="utf-8"))["modules"]] == ["jockey_course", "graded_picks"]
    assert json.loads((ra / "Z1.json").read_text(encoding="utf-8"))["modules"] == [other]


def test_module_not_applicable_without_history(tmp_path, monkeypatch):
    from src.research.race_analysis.base import RaceContext
    monkeypatch.setenv("GRADED_PICKS_DIR", str(tmp_path / "none"))
    ctx = RaceContext(race_id="x", date="2026-10-25", venue="京都", surface="芝", distance=3000, race_name="菊花賞")
    assert GP.GradedPicksModule().applicable(ctx) is False


def test_history_accepts_races_after_freeze():
    """凍結の後の回（前向きの記録）を含む表でも、照合（凍結の前の回だけ）で止まらず、その回も記録に入る。"""
    t = _table()
    rep, rules = _cycle(t)
    src = t.r[t.r["race_id"] == "R2026"]
    add = src.assign(race_id="R2026b", dt=src["dt"] + (pd.Timestamp("2026-12-01") - src["dt"].iloc[0]), fin=src["fin"][::-1].to_numpy(), row=src["row"] + 10000)
    r2 = pd.concat([t.r, add], ignore_index=True)
    F2 = np.vstack([t.F, t.F[src.index.to_numpy()]])
    t2 = CY.Table(r2, F2, NAMES)
    h = PK.build_history(t2, _prereg(rules), rep)
    races = {x["race_id"]: x for x in h["rkeys"]["R"]["races"]}
    assert "R2026b" in races and PK.period_of(races["R2026b"]["date"], "2026-11-16") == "forward"


def test_history_is_ignored_when_frozen_rules_differ(tmp_path, monkeypatch):
    h, *_ = _history()
    _write_history(tmp_path, monkeypatch, h)
    assert GP.load_history() is not None
    GP.PREREG_CYCLES.write_text(json.dumps({"sha256_rules": "other", "frozen_at": h["frozen_at"]}), encoding="utf-8")
    assert GP.load_history() is None                # 凍結を作り直した後の古い記録は使わない


def test_compute_failure_is_not_an_error(tmp_path, monkeypatch):
    """計算が失敗しても例外を出さず（status=error ＝欠けた結果にしない）、図なしの注記だけを返す。"""
    from src.research.race_analysis.base import RaceContext
    h, *_ = _history()
    _write_history(tmp_path, monkeypatch, h)
    monkeypatch.setattr(GP, "figure_for", lambda *a, **k: (_ for _ in ()).throw(KeyError("x")))
    out = GP.GradedPicksModule().compute(RaceContext(race_id="R2026", date="2026-05-02", venue=None, surface=None, distance=None, race_name="R"))
    assert out.figure is None and out.view == "graded_picks" and out.notes


def test_default_rules_hide_B_values(tmp_path, monkeypatch):
    """規則が既定の重賞（2018〜2024 年に回が無い）は、B の ◎・消し・成績を出さない（前日版の ◎ だけ）。"""
    h, *_ = _history()
    _write_history(tmp_path, monkeypatch, h)
    v = h["rkeys"]["N"]
    assert v["honmei"]["default"]
    last = max(v["races"], key=lambda x: x["date"])
    f = GP.figure_for(GP.load_history(), "N", last["race_id"], last["date"])
    assert f["this"]["race"]["B"]["honmei"] is None and f["this"]["race"]["base"] is not None
    assert all(x["B"]["honmei"] is None for x in f["past"])
    assert all(s["B_honmei"] is None for s in f["stats"].values())


def test_redact_actual_odds_only_in_graded_tab():
    other = {"id": "jockey_course", "figure": {"odds": 1}}
    fig = {"this": {"race": {"winners": [{"hn": 1, "name": "a", "odds": 3.8, "pop": 1, "fin": 1, "tan": 380, "fuku": 180}]}}}
    p = {"modules": [other, {"id": "graded_picks", "figure": fig}]}
    out = GP.redact_actual_odds(p)
    w = out["modules"][1]["figure"]["this"]["race"]["winners"][0]
    assert w == {"hn": 1, "name": "a", "fin": 1} and out["modules"][0] == other
    assert p["modules"][1]["figure"]["this"]["race"]["winners"][0]["odds"] == 3.8      # 元は書き換えない


def test_api_hides_actual_odds_from_non_developers_in_public_mode(tmp_path, monkeypatch):
    from flask import Flask

    from src.api.v1.routes.race_analysis import bp
    from src.research.race_analysis import store
    from src.utils import public_policy as P

    monkeypatch.setenv("KEIBA_RACE_ANALYSIS_DIR", str(tmp_path / "ra"))
    fig = {"this": {"race": {"winners": [{"hn": 1, "name": "a", "odds": 3.8, "pop": 1, "fin": 1, "tan": 380, "fuku": 180}]}}}
    store.write("202608040911", {"status": "ok", "race": {}, "entries": [], "modules": [{"id": "graded_picks", "figure": fig}], "fingerprint": "f"})
    app = Flask(__name__)
    app.register_blueprint(bp, url_prefix="/api/v1")
    c = app.test_client()
    win = lambda: c.get("/api/v1/races/202608040911/analysis").get_json()["modules"][0]["figure"]["this"]["race"]["winners"][0]   # noqa: E731
    monkeypatch.setattr(P, "is_public_mode", lambda: False)
    assert win()["odds"] == 3.8                                     # private（今）: そのまま
    monkeypatch.setattr(P, "is_public_mode", lambda: True)
    monkeypatch.setattr(P, "actual_odds_allowed", lambda: False)
    monkeypatch.setattr("src.api.flask_auth.is_logged_in", lambda: False)
    assert "odds" not in win() and "tan" not in win() and win()["fin"] == 1
    monkeypatch.setattr("src.api.flask_auth.is_logged_in", lambda: True)
    assert win()["odds"] == 3.8                                     # 開発者には出す
