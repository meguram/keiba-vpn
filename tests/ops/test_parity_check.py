"""並走検証（ホストと GCP の成果物の一致判定）。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.ops.gcs_io import MemoryIO, read_json
from src.scripts.ops import parity_check as P

JST = ZoneInfo("Asia/Tokyo")


def _pq(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_parquet_match_within_tolerance_and_order_independent(tmp_path):
    a = [{"race_id": "r1", "horse_id": "h1", "p_win": 0.2000001}, {"race_id": "r1", "horse_id": "h2", "p_win": 0.8}]
    b = [{"race_id": "r1", "horse_id": "h2", "p_win": 0.8}, {"race_id": "r1", "horse_id": "h1", "p_win": 0.2}]
    _pq(tmp_path / "l/p.parquet", a)
    _pq(tmp_path / "r/p.parquet", b)
    rep = P.compare_dirs("parquet_frame", tmp_path / "l", tmp_path / "r", keys=["race_id", "horse_id"], rtol=1e-4, atol=1e-4)
    assert rep["ok"] and rep["status"] == "match"


def test_parquet_value_key_and_column_differences(tmp_path):
    _pq(tmp_path / "l/p.parquet", [{"k": 1, "v": 1.0, "w": "a"}, {"k": 2, "v": 2.0, "w": "b"}])
    _pq(tmp_path / "r/p.parquet", [{"k": 1, "v": 1.5, "w": "a"}, {"k": 3, "v": 2.0, "w": "b"}])
    rep = P.compare_dirs("parquet_frame", tmp_path / "l", tmp_path / "r", keys=["k"], rtol=1e-6, atol=1e-6)
    assert not rep["ok"]
    text = " ".join(rep["problems"])
    assert "キーの差: 左のみ 1 行 / 右のみ 1 行" in text and "値が許容誤差を超えて違う列" in text
    _pq(tmp_path / "r2/p.parquet", [{"k": 1, "v": 1.0}])
    assert any("列の差" in p for p in P.compare_dirs("parquet_frame", tmp_path / "l", tmp_path / "r2", keys=["k"])["problems"])


def test_json_tree_ignores_volatile_keys_and_float_noise(tmp_path):
    (tmp_path / "l").mkdir()
    (tmp_path / "r").mkdir()
    (tmp_path / "l/a.json").write_text(json.dumps({"x": 1.0000000001, "generated_at": "t1", "n": {"p": [0.5, 0.25]}}))
    (tmp_path / "r/a.json").write_text(json.dumps({"x": 1.0, "generated_at": "t2", "n": {"p": [0.5, 0.25]}}))
    assert P.compare_dirs("json_tree", tmp_path / "l", tmp_path / "r", rtol=1e-6, atol=1e-6)["ok"]
    (tmp_path / "r/a.json").write_text(json.dumps({"x": 1.1, "n": {"p": [0.5]}}))
    rep = P.compare_dirs("json_tree", tmp_path / "l", tmp_path / "r")
    assert not rep["ok"] and any("長さ" in d for d in rep["json_diffs"]) and any("$.x" in d for d in rep["json_diffs"])


def test_file_hash_and_file_set_difference(tmp_path):
    for side, data in (("l", b"1"), ("r", b"2")):
        (tmp_path / side).mkdir()
        (tmp_path / side / "f.bin").write_bytes(data)
    (tmp_path / "r/extra.bin").write_bytes(b"x")
    rep = P.compare_dirs("file_hash", tmp_path / "l", tmp_path / "r")
    assert not rep["ok"] and rep["only_right"] == ["extra.bin"] and any("内容が違う" in p for p in rep["problems"])


def test_both_empty_is_not_a_match(tmp_path):
    (tmp_path / "l").mkdir()
    (tmp_path / "r").mkdir()
    rep = P.compare_dirs("file_hash", tmp_path / "l", tmp_path / "r")
    assert rep["ok"] is False and rep["status"] == "empty"


def _doc():
    return {"jobs": [{"id": "j1", "destination": "gcp", "parity": {"kind": "json_tree", "paths": ["data/o/{today_iso}"], "keys": [], "rtol": 1e-6, "atol": 1e-6}},
                     {"id": "j2", "destination": "gcp", "parity": {"kind": "none"}}]}


def test_auto_compares_host_snapshot_with_gcp_state_and_writes_reports():
    io = MemoryIO()
    io.put_bytes("host_snapshot/20261024/data/o/2026-10-24/a.json", json.dumps({"v": 1}).encode())
    io.put_bytes("state/data/o/2026-10-24/a.json", json.dumps({"v": 1}).encode())
    sent = []
    import os
    os.environ["KEIBA_SLACK_NOTIFY"] = "1"
    try:
        out = P.run_auto(io, "20261024", None, write=True, send=True, now=datetime(2026, 10, 24, 8, 0, tzinfo=JST), doc=_doc(), send_fn=lambda payload, channel: sent.append((payload, channel)) or {"ok": True})
    finally:
        os.environ.pop("KEIBA_SLACK_NOTIFY")
    assert out["reports"] == [{"job": "j1", "status": "match", "ok": True}]
    assert read_json(io, "ops/parity/20261024/j1.json")["ok"] is True
    assert sent and sent[0][1] == "report" and "並走検証" in sent[0][0]["text"]


def test_auto_reports_missing_sides_and_mismatch():
    io = MemoryIO()
    out = P.run_auto(io, "20261024", None, write=False, send=False, doc=_doc())
    assert out["reports"][0]["status"] == "no_data"
    io.put_bytes("host_snapshot/20261024/data/o/2026-10-24/a.json", b'{"v":1}')
    assert P.run_auto(io, "20261024", None, write=False, send=False, doc=_doc())["reports"][0]["status"] == "no_data"      # GCP 側が無い
    io.put_bytes("state/data/o/2026-10-24/a.json", b'{"v":2}')
    assert P.run_auto(io, "20261024", None, write=False, send=False, doc=_doc())["reports"][0]["status"] == "mismatch"


def test_summary_marks_warning_when_any_problem():
    rep = [{"job": "a", "status": "match", "paths": []}, {"job": "b", "status": "mismatch", "paths": [{"problems": ["JSON の差 1 件"]}]}]
    msg = P.build_summary(rep, "20261024")
    assert msg["attachments"][0]["color"] != "#3b82f6" and "1 件に問題" in json.dumps(msg, ensure_ascii=False)


def test_real_job_map_parity_paths_are_declared_in_state_out():
    # 並走検証が見るパスは、そのジョブの state_out(= GCS の state/ に上がる場所)の下にあること
    from src.ops import job_map

    for j in job_map.by_destination("gcp", job_map.load()):
        par = j.get("parity") or {}
        if par.get("kind", "none") == "none":
            continue
        outs = [o["path"].rstrip("/") for o in j.get("state_out") or []]
        for p in par["paths"]:
            base = p.split("{")[0].rstrip("/") if "{" in p else p
            assert any(base == o or base.startswith(o + "/") or o.startswith(base + "/") or o == base for o in outs), f"{j['id']}: parity {p} が state_out {outs} の下に無い"
