"""重賞の「データで予想」の前日の計算（src/research/graded/upcoming.py）と、その周り（folklore の出馬表の行・frame の読み込み・
entries-sync のつなぎ・タブの表示）のテスト。重い計算（folklore・frame・tally）は流さない（実データでの確かめは --sim）。"""

from __future__ import annotations

import json
import os
import time

import pandas as pd
import pytest

from src.research.graded import upcoming as UP
from src.research.race_analysis import graded_picks as GP


def _card(rid="202605050811", name="R", date="2026-05-02", nums=(1, 2, 3, 4, 5)):
    return {"race_id": rid, "race_name": name, "date": date, "venue": "東京", "surface": "芝", "distance": 1600, "direction": "左",
            "race_class": "サラ系３歳", "start_time": "15:40", "round": 11, "weight_rule": "定量", "grade": "OP",
            "entries": [{"horse_number": n, "bracket_number": n, "horse_name": f"馬{n}", "horse_id": f"h{n}", "jockey_id": "j", "jockey_name": "騎",
                         "trainer_id": "t", "sex_age": "牡3", "jockey_weight": 57.0, "odds": 3.5, "popularity": 1, "weight": 480, "weight_change": 2}
                        for n in nums]}


def _history(tmp_path, monkeypatch, rkeys=("R",)):
    d = tmp_path / "gp"
    d.mkdir(exist_ok=True)
    h = {"rkeys": {k: {"rkey": k, "grade": "G2", "course": "c", "honmei": {"default": False, "dayof": False, "rule": {}, "label": ""},
                       "kill": None, "races": []} for k in rkeys},
         "sha256_rules": "s", "frozen_at": "f", "track_from": "2026-11-16", "notes": []}
    (d / "history.json").write_text(json.dumps(h), encoding="utf-8")
    pr = tmp_path / "prereg_cycles.json"
    pr.write_text(json.dumps({"sha256_rules": "s", "frozen_at": "f"}), encoding="utf-8")
    monkeypatch.setattr(GP, "PREREG_CYCLES", pr)
    monkeypatch.setenv("GRADED_PICKS_DIR", str(d))
    monkeypatch.setattr("src.research.graded.race_keys.canonical_key", lambda name, month=None: name)
    return d, h


def test_confirmed_needs_unique_positive_numbers():
    assert UP.confirmed(_card())
    assert not UP.confirmed(_card(nums=(0, 0, 0)))          # 特別登録の段階（馬番なし）
    assert not UP.confirmed(_card(nums=(1, 1, 2)))
    assert not UP.confirmed({"entries": []})


def test_rows_from_card_carry_no_result_or_day_of_values():
    rows = UP.rows_from_card(_card(), "G2")
    assert len(rows) == 5 and {r["grade"] for r in rows} == {"G2"} and rows[0]["field_size"] == 5
    for k in ("odds", "popularity", "weight", "weight_change", "finish_position", "track_condition"):
        assert all(k not in r for r in rows)             # オッズ・人気・馬体重・馬場・結果は入れない


class _Storage:
    def __init__(self, cards):
        self.cards = cards

    def load(self, category, key, bypass_cache=False):
        return self.cards.get(key) if category == "race_shutuba" else None


def _plan(*ids, name="R"):
    return [{"race_id": i, "race_name": name if i != "B" else "ふつう", "date": "20260502"} for i in ids]


def test_save_entries_only_confirmed_graded_and_only_when_changed(tmp_path, monkeypatch):
    d, _ = _history(tmp_path, monkeypatch)
    st = _Storage({"A": _card("A"), "B": _card("B", name="ふつう"), "C": _card("C", nums=(0, 0, 0))})
    loaded = []
    orig = st.load
    st.load = lambda c, k, bypass_cache=False: (loaded.append(k), orig(c, k))[1]
    assert UP.save_entries(st, _plan("A", "B", "C")) == ["A"]
    assert "B" not in loaded                                       # 重賞の名前でないレースはカードを読まない
    body = json.loads((d / "entries" / "A.json").read_text(encoding="utf-8"))
    assert body["race_id"] == "A" and set(body["entries"][0]) == set(UP.ENTRY_KEYS)   # オッズ・人気・馬体重は書き置かない
    assert UP.save_entries(st, _plan("A")) == []                   # 中身が同じなら書かない
    st.cards["A"]["entries"][0]["odds"] = 9.9                      # オッズが動いても書き直さない（計算を起こさない）
    assert UP.save_entries(st, _plan("A")) == []
    st.cards["A"]["entries"][0]["jockey_id"] = "j2"
    assert UP.save_entries(st, _plan("A")) == ["A"]                # 乗り替わりなどで変われば書き直す


def test_kick_starts_one_job_uses_pid_lock_and_skips_done(tmp_path, monkeypatch):
    d, _ = _history(tmp_path, monkeypatch)
    calls = []

    class P:
        pid = os.getpid()                                          # 動いている計算に見立てる

    monkeypatch.setattr(UP.subprocess, "Popen", lambda *a, **k: (calls.append(a[0]), P())[1])
    (d / "entries").mkdir()
    (d / "entries" / "A.json").write_text(json.dumps({"date": "2026-05-02"}), encoding="utf-8")
    (d / "entries" / "OLD.json").write_text(json.dumps({"date": "2026-04-01"}), encoding="utf-8")
    assert UP.kick([], today="2026-05-01") == {"kicked": ["A"]}    # 書き置き済みで今日以降の、計算がまだのレース
    assert calls[0][calls[0].index("--races") + 1] == "A"
    assert UP.kick(["A"], today="2026-05-01")["running"] is True   # 動いている間は 2 本目を起動しない
    (d / UP.LOCK).write_text("999999999 x", encoding="utf-8")      # 終わった（落ちた）計算のロックは古いとみなす
    assert UP.kick(["A"], today="2026-05-01") == {"kicked": ["A"]}
    (d / "races").mkdir()
    (d / "races" / "A.json").write_text(json.dumps({"entries_sig": UP.entries_sig("A")}), encoding="utf-8")
    assert UP.kick(["A"], today="2026-05-01") == {"kicked": []}    # 同じ出馬表で計算済み
    (d / "entries" / "A.json").write_text(json.dumps({"date": "2026-05-02", "x": 1}), encoding="utf-8")
    assert UP.stale("A")                                            # 出馬表が変われば計算し直す


def test_entries_sync_hook_never_raises(monkeypatch):
    from src.scraper import entries_sync as ES
    monkeypatch.setattr(UP, "save_entries", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    r = ES._kick_graded_picks(None, ["A"])
    assert "error" in r and "seconds" in r


def test_figure_shows_picked_only_with_same_frozen_rules(tmp_path, monkeypatch):
    d, h = _history(tmp_path, monkeypatch)
    (d / "races").mkdir()
    pk = {"race_id": "X1", "rkey": "R", "date": "2026-12-01", "data_until": "2026-11-30", "computed_at": "2026-11-30T19:00:00+09:00", "field": 5,
          "sha256_rules": "s", "frozen_at": "f", "base": {"hn": 1, "name": "a"}, "B": {"honmei": {"hn": 2, "name": "b"}, "kill": [{"hn": 5, "name": "e"}]},
          "skipped": {}}
    (d / "races" / "X1.json").write_text(json.dumps(pk), encoding="utf-8")
    f = GP.figure_for(GP.load_history(), "R", "X1", "2026-12-01")
    assert f["this"]["status"] == "picked" and f["this"]["picked"]["B"]["honmei"]["hn"] == 2
    (d / "races" / "X1.json").write_text(json.dumps(pk | {"sha256_rules": "old"}), encoding="utf-8")
    assert GP.figure_for(GP.load_history(), "R", "X1", "2026-12-01")["this"]["status"] == "pending"
    # 今の出馬表と違う出馬表で計算したもの（取消・乗り替わりの前）は出さない
    (d / "entries").mkdir()
    (d / "entries" / "X1.json").write_text("{}", encoding="utf-8")
    (d / "races" / "X1.json").write_text(json.dumps(pk | {"entries_sig": "old"}), encoding="utf-8")
    assert GP.figure_for(GP.load_history(), "R", "X1", "2026-12-01")["this"]["status"] == "pending"
    (d / "races" / "X1.json").write_text(json.dumps(pk | {"entries_sig": UP.entries_sig("X1")}), encoding="utf-8")
    assert GP.figure_for(GP.load_history(), "R", "X1", "2026-12-01")["this"]["status"] == "picked"
    # 消しだけが既定の重賞は、消しを出さない（◎ は出す）
    h2 = json.loads(json.dumps(h))
    h2["rkeys"]["R"]["kill"] = {"default": True, "dayof": False}
    p2 = GP.load_picked("X1", h2)
    assert p2["B"]["kill"] is None and p2["B"]["honmei"]["hn"] == 2 and p2["skipped"]["kill"] == "default"


def test_master_entry_rows_start_and_are_not_treated_as_dnf(tmp_path, monkeypatch):
    """frame の読み込み: 出馬表の行は発走する前提で残し、中止の馬（着外として扱い直す）とは分ける。"""
    from src.research.graded import features as F
    base = {c: None for c in F.MASTER_COLS}
    d = pd.DataFrame([
        base | {"race_id": "1", "horse_id": "a", "dt": "2026-01-01", "p_dt": None, "fin": 1.0, "surf": "芝", "ok": "True", "y3": 1.0, "y1": 1.0},
        base | {"race_id": "1", "horse_id": "b", "dt": "2026-01-01", "p_dt": None, "fin": None, "surf": "芝", "ok": "False", "y3": None, "y1": None},
        base | {"race_id": "2", "horse_id": "a", "dt": "2026-02-01", "p_dt": None, "fin": None, "surf": "芝", "ok": "True", "y3": None, "y1": None},
    ])
    d["odds"] = [2.0, 5.0, None]
    d["is_entry"] = [False, False, True]
    p = tmp_path / "m.parquet"
    d.to_parquet(p)
    m = F.load_master(p)
    assert m["started"].tolist() == [True, True, True]
    assert m["ok"].tolist() == [True, True, True]
    assert m["is_entry"].tolist() == [False, False, True]
    assert pd.isna(m["y3"].iloc[2])


def test_folklore_entries_and_cutoff(tmp_path, monkeypatch):
    """folklore の結果の読み込み: CUTOFF の日以降の結果を捨て、出馬表の行を足す（結果・当日の値は空にする）。"""
    import importlib
    monkeypatch.setenv("FOLKLORE_YEAR_END", "2026")
    ep = tmp_path / "e.parquet"
    pd.DataFrame([{"race_id": "202605050811", "date": "2026-05-03", "horse_id": "h1", "horse_number": 1, "odds": 3.0, "weight": 480,
                   "finish_position": 1}]).to_parquet(ep)
    monkeypatch.setenv("FOLKLORE_ENTRIES", str(ep))
    monkeypatch.setenv("FOLKLORE_CUTOFF", "2026-05-03")
    import src.scripts.data.folklore_build as FB
    FB = importlib.reload(FB)
    cols = ["race_id", "date", "venue", "venue_code", "surface", "distance", "direction", "grade", "race_class", "weather", "track_condition",
            "start_time", "field_size", "race_name", "round", "pace", "lap_times", "finish_position", "bracket_number", "horse_number", "horse_id",
            "sex_age", "jockey_weight", "jockey_name", "jockey_id", "time_sec", "passing_order", "last_3f", "odds", "popularity", "weight",
            "weight_change", "trainer_id"]
    res = pd.DataFrame([{c: None for c in cols} | {"race_id": r, "date": dt, "horse_id": "h1", "finish_position": 2} for r, dt in
                        (("202605050711", "2026-05-02"), ("202605050712", "2026-05-03"))])
    monkeypatch.setattr(FB.pd, "read_parquet", lambda f, columns=None: res[columns] if "race_result_flat" in str(f) else _RP(f, columns=columns))
    monkeypatch.setattr(FB, "read_years", lambda name, years, cols: pd.DataFrame({"race_id": ["202605050711"], "horse_id": ["h1"], "weight_rule": ["定量"]}))
    monkeypatch.setattr(FB, "YEARS", [2026])
    d = FB.load_results()
    assert sorted(d["race_id"]) == ["202605050711", "202605050811"]          # 5/3 の結果は捨て、出馬表の行を足した
    e = d[d["is_entry"]].iloc[0]
    assert pd.isna(e["odds"]) and pd.isna(e["weight"]) and pd.isna(e["finish_position"])
    importlib.reload(FB)


_RP = pd.read_parquet


@pytest.fixture(autouse=True)
def _reload_folklore_after(monkeypatch):
    yield
    import importlib

    import src.scripts.data.folklore_build as FB
    for k in ("FOLKLORE_ENTRIES", "FOLKLORE_CUTOFF"):
        os.environ.pop(k, None)
    importlib.reload(FB)
