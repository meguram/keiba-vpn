"""候補の影のランナー candidate_runner の判断（純粋関数）と、T-60 オッズの手元の写し odds_t60_copy。重い推論は動かさない。外部接続なし。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.pipeline.inference import candidate_runner as cr
from src.pipeline.inference import odds_t60_copy as oc

JST = timezone(timedelta(hours=9))
POST = datetime(2026, 11, 21, 10, 0, tzinfo=JST)
RACES = [{"race_id": "A", "post_time": POST}]


def _plan(now_min_before, st):
    return cr.plan_actions(POST - timedelta(minutes=now_min_before), RACES, {"A": st}, deadline_min=20, predict_after_min=30)


def test_predict_after_main_or_t30():
    assert _plan(55, {})["predict"] == []
    assert _plan(55, {"main_done": True})["predict"] == ["A"]
    assert _plan(29, {})["predict"] == ["A"]                       # 本体が遅れても T-30 を過ぎたら流す
    assert _plan(29, {"failed": True})["predict"] == []            # 失敗したレースは流し直さない（期限で見送り）


def test_stage2_when_predicted_and_odds():
    assert _plan(50, {"predicted": True, "odds": {"x": 1}})["stage2"] == ["A"]
    assert _plan(50, {"predicted": True})["stage2"] == []


def test_expire_at_deadline_and_reason():
    assert _plan(21, {"predicted": True})["expire"] == []
    assert _plan(20, {"predicted": True})["expire"] == ["A"]
    assert cr.expire_reason({"predicted": True}) == "no_odds_t60"
    assert cr.expire_reason({"failed": True, "odds": {}}) == "no_prediction"
    assert cr.expire_reason({}) == "no_prediction"


def test_nothing_after_post_or_when_decided():
    assert _plan(-1, {"predicted": True, "odds": {"x": 1}}) == {"predict": [], "stage2": [], "expire": []}
    assert _plan(10, {"decided": True}) == {"predict": [], "stage2": [], "expire": []}


def test_runners_are_horses_with_odds():
    run, od = cr.runners_and_odds([1, 2, 3, 4], [{"horse_number": 1, "win_odds": 3.2}, {"horse_number": 2, "win_odds": 0},
                                                  {"horse_number": 3, "win_odds": None}, {"horse_number": 4, "win_odds": 7}])
    assert run == [1, 4] and od == {1: 3.2, 4: 7.0}


def test_main_done_ids(tmp_path):
    p = tmp_path / "20261121.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in [{"race_id": "A", "status": "ok"}, {"race_id": "B", "status": "error"}]) + "\nbroken\n", encoding="utf-8")
    assert cr.main_done_ids(p) == {"A"}
    assert cr.main_done_ids(tmp_path / "none.jsonl") == set()


def test_config_is_off_by_default_and_complete():
    cfg = cr.load_config()
    assert cfg["enabled"] is False
    env = cfg["env"]
    assert env["KEIBA_FIRST_YEAR"] == "2016" and env["KEIBA_INFERENCE_ENGINE"] == "full" and env["KEIBA_ODDS_SEC77"] == "0"
    assert cfg["redirect_2016"]["rebuild_base"].endswith("base_2016_fix2")
    assert not cfg["env"]["KEIBA_INFERENCE_DIR"].startswith(cfg["main_inference_dir"])   # 影の作業場所は本番と別


def test_odds_copy_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("KEIBA_ODDS_T60_LOCAL_COPY", raising=False)
    monkeypatch.setenv("KEIBA_ODDS_T60_DIR", str(tmp_path))
    doc = {"snapshot_at": "2026-11-21T09:00:05+09:00", "entries": [{"horse_number": 1, "win_odds": 2.3, "popularity": 1}]}
    assert oc.write_copy("R1", doc, "t60") is None and not any(tmp_path.iterdir())


def test_odds_copy_t60_only_and_readable(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_ODDS_T60_LOCAL_COPY", "1")
    monkeypatch.setenv("KEIBA_ODDS_T60_DIR", str(tmp_path))
    doc = {"snapshot_at": "2026-11-21T09:00:05+09:00", "entries": [{"horse_number": 1, "win_odds": 2.3, "popularity": 1}, {"horse_number": 0}]}
    assert oc.write_copy("R1", doc, "t15") is None
    p = oc.write_copy("R1", doc, "t60")
    assert p == tmp_path / "20261121" / "R1.json"
    got = oc.read_copy("20261121", "R1")
    assert got["entries"] == [{"horse_number": 1, "win_odds": 2.3, "popularity": 1}] and got["snapshot_at"] == doc["snapshot_at"]


def test_odds_copy_failure_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_ODDS_T60_LOCAL_COPY", "1")
    f = tmp_path / "file"
    f.write_text("x")
    monkeypatch.setenv("KEIBA_ODDS_T60_DIR", str(f))          # フォルダを作れない
    assert oc.write_copy("R1", {"entries": []}, "t60") is None


def test_set_disabled_writes_mark_and_touches_decisions(tmp_path):
    cfg = {"publish_dir": str(tmp_path), "watch_dir": str(tmp_path / "w")}
    d = tmp_path / "bets_stage2" / "20261121"
    d.mkdir(parents=True)
    (d / "R1.json").write_text(json.dumps({"status": "buy"}), encoding="utf-8")
    out = cr.set_disabled(cfg, True, "20261121")
    assert out["touched"] == 1 and json.loads((tmp_path / "bets_stage2" / "_disabled.json").read_text())["disabled"] is True
    assert "toggled_at" in json.loads((d / "R1.json").read_text())
    cr.set_disabled(cfg, False, "20261121")
    assert json.loads((tmp_path / "bets_stage2" / "_disabled.json").read_text())["disabled"] is False
    cr.set_disabled(cfg, True, "20261121", v6c_only=True)
    assert (tmp_path / "predictions_v6c" / "_disabled.json").exists()


def _bare_runner(tmp_path):
    r = cr.Runner.__new__(cr.Runner)
    r.cfg, r.day, r.replay = {}, "20261121", False
    r.publish, r.watch_dir = tmp_path / "pub", tmp_path / "w"
    r.st, r.post = {"A": {}}, {"A": POST}
    r.now_fn = lambda: POST - timedelta(minutes=30)
    r.frozen = {"hash": "70ba29fa7b55"}
    return r


def test_system_failures_write_no_decision(tmp_path):
    """システムの原因（T-60 の写しが無い・推論の失敗など）の見送りは判定ファイルを書かない → API は今の target のまま。"""
    r = _bare_runner(tmp_path)
    r.st["A"] = {"predicted": True}
    r.expire("A")
    assert r.st["A"]["decided"] and not (tmp_path / "pub").exists()
    rows = [json.loads(x) for x in (tmp_path / "w" / "20261121.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["event"] == "system_skip" and rows[-1]["reason"] == "no_odds_t60"
    assert set(cr.SYSTEM_REASONS) >= {"no_prediction", "no_odds_t60", "stage2_error", "odds_broken", "no_stage_inputs"}


def test_rule_decision_written_once_and_not_after_post(tmp_path):
    r = _bare_runner(tmp_path)
    r._decide("A", {"strategy": "stage2_r2", "status": "skip", "skip_reason": "r_max"}, 0.0)
    p = tmp_path / "pub" / "bets_stage2" / "20261121" / "A.json"
    assert json.loads(p.read_text())["skip_reason"] == "r_max"
    r.st["A"]["decided"] = False
    r._decide("A", {"strategy": "stage2_r2", "status": "buy"}, 0.0)                # 2 回目は書かない
    assert json.loads(p.read_text())["skip_reason"] == "r_max"
    r2 = _bare_runner(tmp_path / "late")
    r2.now_fn = lambda: POST + timedelta(minutes=1)
    r2._decide("A", {"strategy": "stage2_r2", "status": "buy"}, 0.0)                # 発走後は書かない
    assert not (tmp_path / "late" / "pub").exists()


def test_enable_mark(tmp_path, monkeypatch):
    monkeypatch.delenv("KEIBA_CANDIDATE_RUNNER", raising=False)
    monkeypatch.setattr(cr, "ENABLE_MARK", tmp_path / "candidate_runner.on")
    assert not cr.enabled({"enabled": False})
    (tmp_path / "candidate_runner.on").write_text("x")
    assert cr.enabled({"enabled": False})


def test_no_stage2_after_deadline():
    """期限（T-20）を過ぎてからは、材料がそろっても判定しない（システムの原因の見送り＝今の表示）。"""
    assert _plan(19, {"predicted": True, "odds": {"x": 1}}) == {"predict": [], "stage2": [], "expire": ["A"]}


class _FakeS2:
    """第 2 段のモデルの代わり: 渡されたオッズをそのまま記録し、q と offset がそのオッズから作られることを確かめる。"""
    name = "fake"

    def __init__(self):
        self.seen = None

    def predict_race(self, X, odds):
        import numpy as np
        import pandas as pd

        from src.pipeline.inference.stage2_market import market_q

        self.seen = np.asarray(odds, dtype=float)
        q = market_q(self.seen)
        p = q.copy()
        p[2] = p[2] * 4
        p = p / p.sum()
        return pd.DataFrame({"p_win": p, "p_top2": 2 * p, "p_top3": 3 * p, "q": q, "r": np.log(p / q), "odds": self.seen})


def _stage2_runner(tmp_path):
    import pandas as pd

    from src.pipeline.inference import stage2_bets as sb

    r = _bare_runner(tmp_path)
    r.frozen = sb.load_frozen()
    r.s2, r.bundle_name = _FakeS2(), "b"
    r.X = {"A": pd.DataFrame({"race_id": "A", "horse_id": [f"h{i}" for i in range(8)], "horse_number": list(range(1, 9))})}
    r.card = {}
    r._grade = lambda rid: "未勝利"
    return r


def test_stage2_uses_only_t60_odds_for_q_and_offset(tmp_path):
    """q・offset（第 2 段のモデルへの入力）と人気は、T-60 の写しのオッズだけから作る。判定ファイルに odds_phase=t60 が残る。"""
    r = _stage2_runner(tmp_path)
    odds = [2.0, 3.0, 9.0, 7.0, 8.0, 12.0, 20.0, 40.0]
    doc = {"phase": "t60", "snapshot_at": "2026-11-21T09:00:05+09:00",
           "entries": [{"horse_number": i + 1, "win_odds": o, "popularity": 0} for i, o in enumerate(odds)]}
    r.stage2("A", doc)
    assert list(r.s2.seen) == odds
    d = json.loads((tmp_path / "pub" / "bets_stage2" / "20261121" / "A.json").read_text(encoding="utf-8"))
    assert d["odds_phase"] == "t60" and d["odds_t60_at"] == doc["snapshot_at"]
    assert d["horses"][0]["odds"] in odds                   # 判定に使ったオッズも T-60 のもの


def test_stage2_refuses_non_t60_odds(tmp_path):
    r = _stage2_runner(tmp_path)
    r.stage2("A", {"phase": "final", "entries": [{"horse_number": 1, "win_odds": 2.0}]})
    assert r.s2.seen is None and not (tmp_path / "pub").exists()       # 確定・直前のオッズでは判定しない（システムの原因の見送り）


def test_preflight_lists_missing_workspace_files(tmp_path):
    """影の作業場所に必須のものが無ければ一覧に出る（PX・第 2 段・束・RB が無いと main は止まる＝判定を書かない）。"""
    cfg = {"env": {"KEIBA_INFERENCE_DIR": str(tmp_path / "inf"), "KEIBA_STAGE2_MODEL_DIR": str(tmp_path / "s2"),
                   "KEIBA_V6_BUNDLE": str(tmp_path / "b"), "KEIBA_V6_RB_STATE": str(tmp_path / "rb")}}
    miss = cr.preflight(cfg, "20261121")
    assert any(m.startswith("PX") for m in miss) and any(m.startswith("第 2 段") for m in miss) and any(m.startswith("RB") for m in miss)
    for p in ("inf/state/px_v5/svd.npz", "inf/tracking_v2/2026-11-21/x", "inf/lap_forecast/2026-11-21/x", "s2/meta.json", "b/bundle.json", "rb/params.json"):
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_text("{}")
    assert cr.preflight(cfg, "20261121") == []


def test_freshness_stops_on_old_rb_state(tmp_path):
    """RB の状態の cut が履歴の表の最終日より古いと止める（warm の後に StaleInputs → 判定を書かない＝今の表示）。"""
    import pandas as pd

    r = _bare_runner(tmp_path)
    r.ctx = type("C", (), {"history_last_date": "2026-11-16"})()
    rb = tmp_path / "rb"
    rb.mkdir()
    pd.DataFrame({"date": pd.to_datetime(["2026-11-15"]), "rb_mg_x": [1.0]}).to_parquet(rb / "hist.parquet")
    (rb / "params.json").write_text(json.dumps({"cut": "2026-11-15"}))
    assert any("RB の状態が古い" in m for m in r.freshness({"rb_state": str(rb)}))
    (rb / "params.json").write_text(json.dumps({"cut": "2026-11-16"}))
    assert r.freshness({"rb_state": str(rb)}) == []
