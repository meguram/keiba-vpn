"""無料ユーザーに出す印（public_mark_type）の決め方（src/research/public_marks/marks.py）。"""
import json

import numpy as np
import pytest

from src.research.public_marks import marks as M


def _race(n=10):
    hn = list(range(1, n + 1))
    pw = np.array([0.30, 0.20, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02][:n])
    odds = np.array([2.5, 6.0, 4.0, 3.5, 25.0, 30.0, 40.0, 60.0, 80.0, 120.0][:n])
    pop = np.argsort(np.argsort(odds)) + 1
    p3 = np.array([0.6, 0.45, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05][:n])
    return hn, pw, p3, odds, pop


def test_harville_top3_sums_to_three_and_is_monotone():
    pw = np.array([0.4, 0.3, 0.2, 0.1])
    t3 = M.harville_top3(pw)
    assert abs(t3.sum() - 3.0) < 1e-9 and np.all(np.diff(t3) <= 0)


def test_decide_assigns_marks_by_ai_evaluation_not_by_popularity():
    hn, pw, p3, odds, pop = _race()
    r = M.decide(hn, pw, p3, odds, pop)
    m = r["marks"]
    assert r["no_mark"] is None
    assert m[1] == "honmei" and m[2] == "pair"                     # p_win の順
    assert m[3] == "show_val"                                        # ◎○ を除いた Harville の 3 着内の最大
    assert m[4] == "anchor"                                          # p の順位 5 位以内・人気 3 位以内
    q = (1 / odds) / (1 / odds).sum()
    r = np.log((pw / pw.sum()) / q)
    cand = [i for i in range(10) if odds[i] >= 10 and r[i] > 0 and (i + 1) not in (1, 2, 3, 4)]
    star = max(cand, key=lambda i: r[i]) + 1
    assert m[star] == "star" and sum(v == "star" for v in m.values()) == 1   # 10 倍以上で r = log(p/q) > 0 の最大の 1 頭
    assert all(v != "kesi" or pop[k - 1] >= 10 for k, v in m.items())


def test_no_mark_rules_field_odds_split():
    hn, pw, p3, odds, pop = _race(7)
    assert M.decide(hn, pw / pw.sum(), p3, odds, pop)["no_mark"] == "field"
    hn, pw, p3, odds, pop = _race()
    odds2 = odds.copy()
    odds2[3] = np.nan
    assert M.decide(hn, pw, p3, odds2, pop)["no_mark"] == "odds"
    flat = np.full(10, 0.1)
    assert M.decide(hn, flat, p3, odds, pop)["no_mark"] == "split"


def test_decide_does_not_read_results_or_final_odds():
    """入力は p_win・p_top3・T-60 のオッズと人気だけ（関数の引数に着順・確定オッズが無い）。"""
    import inspect

    assert list(inspect.signature(M.decide).parameters) == ["hn", "p_win", "p_top3", "odds", "pop"]


def test_stop_rule_simulation_meets_the_false_stop_limit():
    stop = json.loads(M.STOP.read_text(encoding="utf-8"))
    at = [c["at_races"] for c in stop["checks"]]
    q = np.full(70, 0.27)
    assert M.simulate(q, 1.1, at, n=4000) <= 0.15                   # 決定: 真の比 1.1 で誤って止める確率 15% 以下
    assert M.simulate(q, 0.8, at, n=4000) >= 0.8


def test_config_lists_at_most_two_candidates_per_mark_and_ten_in_all():
    c = json.loads(M.CFG.read_text(encoding="utf-8"))
    lst = c["candidates"]["list"]
    assert len(lst) <= 10 and len(lst) == c["candidates"]["count"]
    per = {}
    for x in lst:
        per[x.split(".")[0]] = per.get(x.split(".")[0], 0) + 1
    assert max(per.values()) <= 2
    assert c["no_mark"]["target_share"] == [0.20, 0.40]
    assert len(c["no_mark"]["rules"]) == 3


def test_shadow_refuses_days_on_or_before_the_freeze(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "_frozen_date", lambda: "20261127")
    with pytest.raises(SystemExit):
        M.shadow_day("20261127", inf=tmp_path, out=tmp_path)


def test_shadow_refuses_before_17_empty_inputs_and_existing_days(tmp_path):
    from datetime import datetime

    late = datetime(2026, 11, 29, 18, 0, tzinfo=M.JST)
    with pytest.raises(SystemExit, match="17:00"):
        M.shadow_day("20261129", inf=tmp_path, out=tmp_path, check_frozen=False, card={"r": {}}, now=datetime(2026, 11, 29, 16, 0, tzinfo=M.JST))
    with pytest.raises(SystemExit, match="出馬表"):
        M.shadow_day("20261129", inf=tmp_path, out=tmp_path, check_frozen=False, card={}, now=late)
    with pytest.raises(SystemExit, match="odds_t60"):
        M.shadow_day("20261129", inf=tmp_path, out=tmp_path, check_frozen=False, card={"r": {}}, now=late)
    (tmp_path / "20261129.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="書き直さない"):
        M.shadow_day("20261129", inf=tmp_path, out=tmp_path, check_frozen=False, card={"r": {}}, now=late)



def test_decide_race_uses_t50_and_separates_timing_reasons():
    """T-50 で決める。オッズの取得・予測の最後の書き込みが T-50 より後なら、時刻の都合の印なし（wait_odds・wait_pred）。"""
    from datetime import datetime, timedelta, timezone

    import pandas as pd

    jst = timezone(timedelta(hours=9))
    post = datetime(2026, 11, 29, 12, 0, tzinfo=jst)
    hn, pw, p3, odds, pop = _race()
    p = pd.DataFrame({"horse_number": hn, "p_win": pw, "p_top3": p3})
    o = {"phase": "t60", "snapshot_at": (post - timedelta(minutes=60)).isoformat(), "written_at": (post - timedelta(minutes=59)).isoformat(),
         "entries": [{"horse_number": h, "win_odds": float(x), "popularity": int(k)} for h, x, k in zip(hn, odds, pop)]}
    ok = M.decide_race(o, p, post, post - timedelta(minutes=55))
    assert ok["no_mark"] is None and ok["decide_at"] == (post - timedelta(minutes=50)).isoformat(timespec="seconds")
    late_odds = {**o, "snapshot_at": (post - timedelta(minutes=45)).isoformat()}
    assert M.decide_race(late_odds, p, post, post - timedelta(minutes=55))["no_mark"] == "wait_odds"
    late_write = {**o, "written_at": (post - timedelta(minutes=48)).isoformat()}           # 取得は T-60 でも、手元に書かれたのが T-50 より後
    assert M.decide_race(late_write, p, post, post - timedelta(minutes=55))["no_mark"] == "wait_odds"
    no_write = {k: v for k, v in o.items() if k != "written_at"}
    assert M.decide_race(no_write, p, post, post - timedelta(minutes=55))["no_mark"] == "odds"
    old = {**o, "snapshot_at": (post - timedelta(minutes=90)).isoformat()}
    assert M.decide_race(old, p, post, post - timedelta(minutes=55))["no_mark"] == "odds"
    assert M.decide_race(o, p, post, post - timedelta(minutes=49))["no_mark"] == "wait_pred"
    assert M.decide_race({**o, "phase": "t15"}, p, post, post - timedelta(minutes=55))["no_mark"] == "odds"
    assert set(M.TIMING) == {"wait_odds", "wait_pred"}


def test_config_records_that_p_does_not_use_odds_and_t50():
    c = json.loads(M.CFG.read_text(encoding="utf-8"))
    assert c["decide"]["p_uses_odds"] is False and "50 分前" in c["decide"]["before_text"]
    assert {r["id"] for r in c["no_mark"]["timing_rules"]} == set(M.TIMING)



def test_harville_top3_matches_brute_force():
    import itertools

    pw = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
    bf = np.zeros(5)
    for a, b, c in itertools.permutations(range(5), 3):
        pr = pw[a] * pw[b] / (1 - pw[a]) * pw[c] / (1 - pw[a] - pw[b])
        for x in (a, b, c):
            bf[x] += pr
    assert np.allclose(M.harville_top3(pw), bf)


def test_scratched_horse_gets_no_mark_and_missing_popularity_blocks_kesi():
    """T-60 の単勝 999.9 の馬は取消として外し、印を付けない（★ にならない）。人気が欠けたら消を付けない。"""
    from datetime import datetime, timedelta, timezone

    import pandas as pd

    post = datetime(2026, 11, 29, 12, 0, tzinfo=timezone(timedelta(hours=9)))
    hn, pw, p3, odds, pop = _race()
    odds = odds.copy()
    odds[9] = 999.9
    p = pd.DataFrame({"horse_number": hn, "p_win": pw, "p_top3": p3})
    o = {"phase": "t60", "snapshot_at": (post - timedelta(minutes=60)).isoformat(), "written_at": (post - timedelta(minutes=59)).isoformat(),
         "entries": [{"horse_number": h, "win_odds": float(x), "popularity": int(k)} for h, x, k in zip(hn, odds, pop)]}
    r = M.decide_race(o, p, post, post - timedelta(minutes=55))
    assert r["scratched"] == [10] and "10" not in {str(k) for k in r["marks"]}
    r_nopred = M.decide_race(o, p[p.horse_number != 10], post, post - timedelta(minutes=55))     # 実データの形: 取消の馬は予測から抜けている
    assert r_nopred["scratched"] == [10] and r_nopred["no_mark"] is None and 10 not in r_nopred["marks"]
    r_missing = M.decide_race(o, p[p.horse_number != 3], post, post - timedelta(minutes=55))     # 取消でない馬が予測にいない → 不整合
    assert r_missing["no_mark"] == "data"
    r2 = M.decide(hn, pw, np.full(10, 0.01), _race()[3], [None] + list(pop[1:]))
    assert "kesi" not in r2["marks"].values()


def test_simulation_with_the_local_q_pool_matches_the_config():
    stop = json.loads(M.STOP.read_text(encoding="utf-8"))
    q = M.honmei_q_pool()
    if len(q) == 0:
        pytest.skip("手元に 11/21〜23 の T-60 と予測が無い（CI）")
    at = [c["at_races"] for c in stop["checks"]]
    assert len(q) == stop["simulation"]["races_in_pool"]
    for k, ratio in (("false_stop_at_ratio_1.1", 1.1), ("stop_at_ratio_1.0", 1.0), ("stop_at_ratio_0.8", 0.8)):
        assert abs(M.simulate(q, ratio, at) - stop["simulation"][k]) < 0.005



def test_shadow_day_writes_rows_from_horse_id_predictions(tmp_path):
    """本番の予測の形（horse_id だけ）を、同じ日の特徴量で馬番に対応づけて行を書く。出馬表から数え、T-60 が無いレースも行にする。"""
    import os
    from datetime import datetime, timedelta

    import pandas as pd

    day = "20261129"
    post = datetime(2026, 11, 29, 12, 0, tzinfo=M.JST)
    hn, pw, p3, odds, pop = _race()
    for sub in ("odds_t60", "predictions_v3", "features"):
        (tmp_path / sub / day).mkdir(parents=True)
    o = {"race_id": "R1", "phase": "t60", "snapshot_at": (post - timedelta(minutes=60)).isoformat(), "written_at": (post - timedelta(minutes=59)).isoformat(),
         "entries": [{"horse_number": h, "win_odds": float(x), "popularity": int(k)} for h, x, k in zip(hn, odds, pop)]}
    (tmp_path / "odds_t60" / day / "R1.json").write_text(json.dumps(o), encoding="utf-8")
    ids = [f"h{h}" for h in hn]
    pd.DataFrame({"race_id": "R1", "horse_id": ids[::-1], "p_win": pw[::-1], "p_top2": pw[::-1], "p_top3": p3[::-1]}).to_parquet(tmp_path / "predictions_v3" / day / "R1.parquet")
    pd.DataFrame({"horse_id": ids, "horse_number": hn}).to_parquet(tmp_path / "features" / day / "R1.parquet")
    t = (post - timedelta(minutes=55)).timestamp()
    os.utime(tmp_path / "predictions_v3" / day / "R1.parquet", (t, t))
    out = tmp_path / "out"
    r = M.shadow_day(day, inf=tmp_path, out=out, check_frozen=False, card={"R1": {"post": post}, "R2": {"post": post}},
                     now=datetime(2026, 11, 29, 18, tzinfo=M.JST))
    assert r["races"] == 2 and r["by_reason"] == {"marked": 1, "odds": 1}
    rows = [json.loads(x) for x in (out / f"{day}.jsonl").read_text(encoding="utf-8").splitlines()]
    r1 = next(x for x in rows if x["race_id"] == "R1")
    assert r1["marks"]["1"] == "honmei" and r1["pred_mtime"].endswith("+09:00") and r1["inputs_sha"]["pred"] and r1["tool_sha"]
