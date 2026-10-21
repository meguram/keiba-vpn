"""deploy/gcp/netkeiba_smoke の試験ロジックと要約。ネットワークには出ない。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "deploy" / "gcp" / "netkeiba_smoke"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"nk_{name}", D / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


smoke = _load("smoke")
summ = _load("summarize")


def _ok(item):
    return {"i": item["i"], "host": item["host"], "code": 200, "bytes": 1000, "marker_ok": True, "block_word": False, "err": None, "sec": 0.5}


def test_ids_and_plan():
    ids = [l for l in (D / "race_ids.txt").read_text().split() if l]
    assert len(ids) == 30 and all(len(i) == 12 and i.isdigit() for i in ids)
    p = smoke.plan(ids)
    assert p[0]["host"] == "db.netkeiba.com" and p[1]["host"] == "race.netkeiba.com"
    assert "login" not in " ".join(x["url"] for x in p)


def test_all_ok_passes_and_waits_between_requests():
    ids = [str(100000000000 + i) for i in range(30)]
    sleeps, out = [], []
    j = smoke.run(ids, fetch_fn=_ok, sleep_fn=sleeps.append, emit=out.append)
    assert j["verdict"] == "PASS" and j["ok"] == 30 and len(out) == 30
    assert len(sleeps) == 29 and all(4.5 <= s <= 7.0 for s in sleeps)


def test_403_aborts_immediately():
    calls = []

    def f(item):
        calls.append(item["i"])
        r = _ok(item)
        if item["i"] == 3:
            r.update(code=403, marker_ok=False, err="HTTP")
        return r

    j = smoke.run([str(i) for i in range(30)], fetch_fn=f, sleep_fn=lambda s: None, emit=lambda r: None)
    assert calls == [0, 1, 2, 3] and j["verdict"] == "FAIL" and j["n403_429"] == 1
    assert any("中止" in r for r in j["reasons"])


def test_five_consecutive_failures_abort_and_rate_threshold():
    def f(item):
        r = _ok(item)
        if item["i"] >= 2:
            r.update(code=500, marker_ok=False, err="HTTP")
        return r

    j = smoke.run([str(i) for i in range(30)], fetch_fn=f, sleep_fn=lambda s: None, emit=lambda r: None)
    assert j["done"] == 7 and j["verdict"] == "FAIL"
    # 2 件だけ失敗（28/30）は合格、3 件失敗（27/30）は不合格
    def make(nfail):
        def g(item):
            r = _ok(item)
            if item["i"] in (5, 10, 15)[:nfail]:
                r.update(marker_ok=False)
            return r
        return g
    assert smoke.run([str(i) for i in range(30)], fetch_fn=make(2), sleep_fn=lambda s: None, emit=lambda r: None)["verdict"] == "PASS"
    assert smoke.run([str(i) for i in range(30)], fetch_fn=make(3), sleep_fn=lambda s: None, emit=lambda r: None)["verdict"] == "FAIL"


def test_block_word_fails():
    def f(item):
        r = _ok(item)
        r["block_word"] = item["i"] == 0
        return r

    assert smoke.run([str(i) for i in range(30)], fetch_fn=f, sleep_fn=lambda s: None, emit=lambda r: None)["verdict"] == "FAIL"


def _logs(verdict="PASS"):
    recs = [{"event": "start", "egress_ip": "203.0.113.5", "slot": "night", "planned": 2},
            {"event": "request", "i": 0, "host": "db.netkeiba.com", "code": 200, "marker_ok": True, "sec": 1.0},
            {"event": "request", "i": 1, "host": "race.netkeiba.com", "code": 200, "marker_ok": True, "sec": 0.5},
            {"event": "summary", "egress_ip": "203.0.113.5", "verdict": verdict, "ok": 2, "planned": 2, "n403_429": 0, "reasons": [] if verdict == "PASS" else ["x"]}]
    return recs


def test_summarize_all_logging_shapes():
    recs = _logs()
    shapes = [
        "\n".join(json.dumps(r) for r in recs),                                              # 標準出力そのもの
        json.dumps([{"textPayload": json.dumps(r)} for r in recs]),                          # textPayload
        json.dumps([{"jsonPayload": {"message": json.dumps(r)}} for r in recs]),             # jsonPayload.message
        json.dumps([{"jsonPayload": r} for r in recs]),                                      # jsonPayload そのもの
    ]
    for s in shapes:
        body, result = summ.summarize(summ.records(s))
        assert result.startswith("RESULT: PASS") and "203.0.113.5" in body and "| night |" in body


def test_summarize_unknown_without_summary_and_fail():
    body, result = summ.summarize(summ.records("\n".join(json.dumps(r) for r in _logs()[:3])))
    assert result.startswith("RESULT: UNKNOWN")
    assert summ.summarize(summ.records("\n".join(json.dumps(r) for r in _logs("FAIL"))))[1].startswith("RESULT: FAIL")
