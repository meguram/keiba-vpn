"""無料ユーザーに出す印（public_mark_type）の決め方 v2（src/research/public_marks/marks.py、config/public_marks_v2.json）。"""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.research.public_marks import marks as M

JST = timezone(timedelta(hours=9))
POST = datetime(2026, 11, 29, 12, 0, tzinfo=JST)


def _race(n=10):
    hn = list(range(1, n + 1))
    pw = np.array([0.30, 0.20, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02][:n])
    odds = np.array([2.5, 6.0, 4.0, 3.5, 25.0, 30.0, 40.0, 60.0, 80.0, 120.0][:n])
    pop = np.argsort(np.argsort(odds)) + 1
    p3 = np.array([0.6, 0.45, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05][:n])
    return hn, pw, p3, odds, pop


def _odds_doc(hn, odds, pop, snap_min=60, write_min=59):
    return {"phase": "t60", "snapshot_at": (POST - timedelta(minutes=snap_min)).isoformat(), "written_at": (POST - timedelta(minutes=write_min)).isoformat(),
            "entries": [{"horse_number": h, "win_odds": float(x), "popularity": int(k)} for h, x, k in zip(hn, odds, pop)]}


def _pred(hn, pw, p3):
    import pandas as pd

    return pd.DataFrame({"horse_number": hn, "p_win": pw, "p_top3": p3})


def test_harville_top3_matches_brute_force():
    import itertools

    pw = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
    bf = np.zeros(5)
    for a, b, c in itertools.permutations(range(5), 3):
        pr = pw[a] * pw[b] / (1 - pw[a]) * pw[c] / (1 - pw[a] - pw[b])
        for x in (a, b, c):
            bf[x] += pr
    assert np.allclose(M.harville_top3(pw), bf)


def test_every_race_gets_honmei_pair_show_star_without_duplicates():
    """v2: 全レースに ◎○▲★ を各 1 頭（重複なし）。★ は ◎○▲ の後、△ より先に付ける。"""
    hn, pw, p3, odds, pop = _race()
    r = M.decide(hn, pw, p3, odds, pop)
    m = r["marks"]
    assert r["status"] == "marked"
    for mk in ("honmei", "pair", "show_val", "star"):
        assert list(m.values()).count(mk) == 1
    assert m[1] == "honmei" and m[2] == "pair" and m[3] == "show_val"
    q = (1 / odds) / (1 / odds).sum()
    rr = np.log(pw / q)
    cand = [i for i in range(10) if odds[i] >= 10 and rr[i] > 0 and (i + 1) not in (1, 2, 3)]
    assert r["mode"] == "ana" and m[max(cand, key=lambda i: rr[i]) + 1] == "star"
    assert m.get(4) == "anchor"


def test_star_falls_back_to_max_r_when_no_longshot_beats_the_market():
    hn, pw, p3, odds, pop = _race()
    odds2 = np.array([2.5, 6.0, 4.0, 3.5, 5.0, 6.5, 7.0, 8.0, 9.0, 9.5])      # 10 倍以上がいない
    r = M.decide(hn, pw, p3, odds2, np.argsort(np.argsort(odds2)) + 1)
    assert r["mode"] == "fallback_r" and list(r["marks"].values()).count("star") == 1


def test_no_odds_marks_by_p_without_anchor_and_kesi():
    hn, pw, p3, _, _ = _race()
    r = M.decide(hn, pw, np.full(10, 0.01))
    m = r["marks"]
    assert r["mode"] == "no_odds_p" and m[4] == "star"                       # ◎○▲ の後の p_win が最大
    assert "anchor" not in m.values() and "kesi" not in m.values()


def test_small_field_marks_up_to_the_field_size():
    hn, pw, p3, odds, pop = _race(3)
    r = M.decide(hn, pw, p3, odds, pop)
    assert sorted(r["marks"].values()) == ["honmei", "pair", "show_val"]
    r2 = M.decide(hn[:1], pw[:1], p3[:1], odds[:1], pop[:1])
    assert r2["marks"] == {1: "honmei"}


def test_split_is_only_a_label():
    hn, _, p3, odds, pop = _race()
    r = M.decide(hn, np.full(10, 0.1), p3, odds, pop)
    assert r["split"] is True and r["status"] == "marked" and list(r["marks"].values()).count("honmei") == 1


def test_kesi_never_overrides_star_and_needs_full_popularity():
    hn, pw, _, odds, pop = _race()
    r = M.decide(hn, pw, np.full(10, 0.01), odds, pop)
    assert list(r["marks"].values()).count("star") == 1
    assert all(v != "kesi" or pop[k - 1] >= 10 for k, v in r["marks"].items())
    r2 = M.decide(hn, pw, np.full(10, 0.01), odds, [None] + list(pop[1:]))
    assert "kesi" not in r2["marks"].values()


def test_decide_race_t50_timing_and_record_only_statuses():
    hn, pw, p3, odds, pop = _race()
    p = _pred(hn, pw, p3)
    ok = M.decide_race(_odds_doc(hn, odds, pop), p, POST, POST - timedelta(minutes=55))
    assert ok["status"] == "marked" and ok["odds_used"] and ok["decide_at"] == (POST - timedelta(minutes=50)).isoformat(timespec="seconds")
    for doc in (_odds_doc(hn, odds, pop, snap_min=45, write_min=44), _odds_doc(hn, odds, pop, write_min=48),
                {k: v for k, v in _odds_doc(hn, odds, pop).items() if k != "written_at"}, _odds_doc(hn, odds, pop, snap_min=90), None):
        r = M.decide_race(doc, p, POST, POST - timedelta(minutes=55))
        assert r["status"] == "marked" and r["mode"] == "no_odds_p" and not r["odds_used"]      # オッズが T-50 に無ければオッズなしで付ける
    assert M.decide_race(_odds_doc(hn, odds, pop), p, POST, POST - timedelta(minutes=49))["status"] == "wait_pred"
    assert M.decide_race(_odds_doc(hn, odds, pop), None, POST, None)["status"] == "no_pred"
    assert M.decide_race(_odds_doc(hn, odds, pop), p.drop(columns=["horse_number"]), POST, POST - timedelta(minutes=55))["status"] == "data"


def test_scratched_horse_gets_no_mark_and_others_still_marked():
    hn, pw, p3, odds, pop = _race()
    odds = odds.copy()
    odds[9] = 999.9
    p = _pred(hn, pw, p3)
    r = M.decide_race(_odds_doc(hn, odds, pop), p, POST, POST - timedelta(minutes=55))
    assert r["scratched"] == [10] and 10 not in r["marks"] and r["odds_used"]
    r2 = M.decide_race(_odds_doc(hn, odds, pop), p[p.horse_number != 10], POST, POST - timedelta(minutes=55))
    assert r2["scratched"] == [10] and r2["odds_used"] and 10 not in r2["marks"]
    r3 = M.decide_race(_odds_doc(hn, odds, pop), p[p.horse_number != 3], POST, POST - timedelta(minutes=55))   # 取消でない馬が予測にいない
    assert r3["status"] == "data" and r3["marks"] == {}
    extra = _pred(hn + [11], list(pw) + [0.01], list(p3) + [0.01])                                             # 予測にしかいない馬
    assert M.decide_race(_odds_doc(hn, odds, pop), extra, POST, POST - timedelta(minutes=55))["status"] == "data"


def test_late_odds_are_not_used_for_scratches_and_partial_odds_fall_back():
    """T-50 より後に書かれたオッズの 999.9 では取消にしない（禁じた入力）。馬はそろうが一部のオッズが欠けるなら no_odds_p。"""
    hn, pw, p3, odds, pop = _race()
    odds = odds.copy()
    odds[0] = 999.9
    p = _pred(hn, pw, p3)
    late = M.decide_race(_odds_doc(hn, odds, pop, snap_min=45, write_min=44), p, POST, POST - timedelta(minutes=55))
    assert late["scratched"] == [] and late["mode"] == "no_odds_p" and late["marks"][1] == "honmei"
    hn, pw, p3, odds, pop = _race()
    doc = _odds_doc(hn, odds, pop)
    doc["entries"][4]["win_odds"] = None
    part = M.decide_race(doc, p, POST, POST - timedelta(minutes=55))
    assert part["mode"] == "no_odds_p" and part["odds_used"] is False and part["status"] == "marked"


def test_star_comes_before_anchor_and_kesi_never_overrides_star():
    """★ と △ の両方に当たる馬は ★（順 ◎○▲★△）。★ の馬が消の条件に当たっても消にしない。"""
    hn = list(range(1, 11))
    pw = np.array([0.30, 0.20, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02])
    odds = np.array([2.5, 6.0, 4.0, 3.5, 25.0, 30.0, 40.0, 60.0, 80.0, 120.0])
    r_ok = M.decide(hn, pw, np.full(10, 0.5), odds, [1, 2, 3, 2, 9, 9, 9, 9, 9, 9])
    star = [k for k, v in r_ok["marks"].items() if v == "star"]
    assert len(star) == 1 and r_ok["marks"].get(4) == "anchor"            # 4 番は ★ の条件（10 倍以上）に当たらないので △
    pop_star = [1, 2, 3, 9, 9, 9, 9, 9, 9, 9]
    pop_star[star[0] - 1] = 3                                              # ★ の馬を人気 3 位にして △ の条件にも当てる
    r_both = M.decide(hn, pw, np.full(10, 0.5), odds, pop_star)
    assert r_both["marks"][star[0]] == "star"                              # ★ が先に付くので △ にならない
    pw2 = np.array([0.30, 0.20, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02])
    odds2 = np.array([2.5, 6.0, 4.0, 3.5, 5.0, 6.0, 7.0, 8.0, 9.0, 200.0])           # 10 番は人気 10・10 倍以上・r > 0 → ★ の候補で、消の条件にも当たる
    pop2 = np.argsort(np.argsort(odds2)) + 1
    q = (1 / odds2) / (1 / odds2).sum()
    assert np.log(pw2[9] / q[9]) > 0
    r2 = M.decide(hn, pw2, np.full(10, 0.01), odds2, pop2)
    assert r2["mode"] == "ana" and r2["marks"][10] == "star"


def test_config_v2_replaces_v1_and_fixes_the_order():
    c = json.loads(M.CFG.read_text(encoding="utf-8"))
    assert c["version"] == 2 and c["replaces"]["hash16"] == "89c2decbbb7af346"
    assert c["order"] == ["honmei", "pair", "show_val", "star", "anchor", "kesi"]
    assert set(c["marks"]["star"]["modes"]) == {"ana", "fallback_r", "no_odds_p"}
    assert c["decide"]["p_uses_odds"] is False and "50 分前" in c["decide"]["before_text"]
    lst = c["candidates"]["list"]
    per = {}
    for x in lst:
        per[x.split(".")[0]] = per.get(x.split(".")[0], 0) + 1
    assert len(lst) <= 10 and max(per.values()) <= 2


def test_stop_rule_simulation_meets_the_false_stop_limit():
    stop = json.loads(M.STOP.read_text(encoding="utf-8"))
    at = [c["at_races"] for c in stop["checks"]]
    q = np.full(70, 0.27)
    assert M.simulate(q, 1.1, at, n=4000) <= 0.15
    assert M.simulate(q, 0.8, at, n=4000) >= 0.8


def test_simulation_with_the_local_q_pool_matches_the_config():
    stop = json.loads(M.STOP.read_text(encoding="utf-8"))
    q = M.honmei_q_pool()
    if len(q) == 0:
        pytest.skip("手元に 11/21〜23 の T-60 と予測が無い（CI）")
    at = [c["at_races"] for c in stop["checks"]]
    assert len(q) == stop["simulation"]["races_in_pool"]
    for k, ratio in (("false_stop_at_ratio_1.1", 1.1), ("stop_at_ratio_1.0", 1.0), ("stop_at_ratio_0.8", 0.8)):
        assert abs(M.simulate(q, ratio, at) - stop["simulation"][k]) < 0.005


def test_shadow_refuses_before_17_empty_inputs_frozen_and_existing_days(monkeypatch, tmp_path):
    late = datetime(2026, 11, 29, 18, 0, tzinfo=JST)
    monkeypatch.setattr(M, "_frozen_date", lambda: "20261129")
    with pytest.raises(SystemExit, match="固めた日"):
        M.shadow_day("20261129", inf=tmp_path, out=tmp_path)
    with pytest.raises(SystemExit, match="17:00"):
        M.shadow_day("20261129", inf=tmp_path, out=tmp_path, check_frozen=False, card={"r": {}}, now=datetime(2026, 11, 29, 16, 0, tzinfo=JST))
    with pytest.raises(SystemExit, match="出馬表"):
        M.shadow_day("20261129", inf=tmp_path, out=tmp_path, check_frozen=False, card={}, now=late)
    with pytest.raises(SystemExit, match="predictions_v3"):
        M.shadow_day("20261129", inf=tmp_path, out=tmp_path, check_frozen=False, card={"r": {}}, now=late)
    (tmp_path / "20261129.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="書き直さない"):
        M.shadow_day("20261129", inf=tmp_path, out=tmp_path, check_frozen=False, card={"r": {}}, now=late)


def test_shadow_day_writes_rows_from_horse_id_predictions(tmp_path):
    import os

    import pandas as pd

    day = "20261129"
    hn, pw, p3, odds, pop = _race()
    for sub in ("odds_t60", "predictions_v3", "features"):
        (tmp_path / sub / day).mkdir(parents=True)
    (tmp_path / "odds_t60" / day / "R1.json").write_text(json.dumps({"race_id": "R1", **_odds_doc(hn, odds, pop)}), encoding="utf-8")
    ids = [f"h{h}" for h in hn]
    pd.DataFrame({"race_id": "R1", "horse_id": ids[::-1], "p_win": pw[::-1], "p_top2": pw[::-1], "p_top3": p3[::-1]}).to_parquet(tmp_path / "predictions_v3" / day / "R1.parquet")
    pd.DataFrame({"horse_id": ids, "horse_number": hn}).to_parquet(tmp_path / "features" / day / "R1.parquet")
    t = (POST - timedelta(minutes=55)).timestamp()
    os.utime(tmp_path / "predictions_v3" / day / "R1.parquet", (t, t))
    out = tmp_path / "out"
    r = M.shadow_day(day, inf=tmp_path, out=out, check_frozen=False, card={"R1": {"post": POST}, "R2": {"post": POST}},
                     now=datetime(2026, 11, 29, 18, tzinfo=JST))
    assert r["races"] == 2 and r["by_status"] == {"marked": 1, "no_pred": 1} and r["by_mode"] == {"ana": 1}
    rows = [json.loads(x) for x in (out / f"{day}.jsonl").read_text(encoding="utf-8").splitlines()]
    r1 = next(x for x in rows if x["race_id"] == "R1")
    assert r1["marks"]["1"] == "honmei" and r1["pred_mtime"].endswith("+09:00") and r1["inputs_sha"]["pred"] and r1["tool_sha"] and r1["mode"] == "ana"
