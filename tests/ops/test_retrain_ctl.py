"""再学習の状態管理 CLI（scripts/dev/retrain_ctl.py）の振る舞い。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CTL = ROOT / "scripts" / "dev" / "retrain_ctl.py"
PLAN = ROOT / "config" / "retrain_plan.json"


def ctl(tmp_path, *args):
    # RETRAIN_ORDER_HTML を存在しない一時ファイルに向ける（既定のままだと、テスト用の状態で本物の retrain_order.html を書き換えてしまう）
    env = {**os.environ, "RETRAIN_STATE_DIR": str(tmp_path), "RETRAIN_PLAN": str(PLAN), "RETRAIN_ORDER_HTML": str(tmp_path / "no_such_order.html")}
    return subprocess.run([sys.executable, str(CTL), *args], env=env, capture_output=True, text=True)


def state(tmp_path):
    return json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))["steps"]


def test_plan_is_consistent():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    ids = [s["id"] for s in plan["steps"]]
    assert len(ids) == len(set(ids)) == 19
    for s in plan["steps"]:
        assert set(s["deps"]) <= set(ids)
        assert ids.index(s["id"]) > max([ids.index(d) for d in s["deps"]] or [-1])
        assert s["commands"] and s["done_check"]
        assert set(s["scopes"]) <= set(plan["scopes"])


def test_default_scope_skips_meguidx_and_optional(tmp_path):
    assert ctl(tmp_path, "status").returncode == 0
    st = state(tmp_path)
    assert st["s1"]["status"] == st["s2"]["status"] == "skipped"
    assert st["s5"]["status"] == st["s6"]["status"] == "skipped"
    assert st["s0"]["status"] == "pending"


def test_dependencies_and_serial_rule(tmp_path):
    ctl(tmp_path, "status")
    assert ctl(tmp_path, "start", "s9").returncode != 0
    assert ctl(tmp_path, "start", "s0").returncode == 0
    assert ctl(tmp_path, "start", "s3").returncode != 0
    assert ctl(tmp_path, "done", "s0", "--note", "ok").returncode == 0
    assert ctl(tmp_path, "start", "s3").returncode == 0
    assert ctl(tmp_path, "start", "s4").returncode != 0


def test_gate_requires_approval(tmp_path):
    ctl(tmp_path, "decide", "scope", "models", "--confirm")
    for sid in ("s0", "s3", "s4", "s7", "s8", "s9", "s10", "s11"):
        assert ctl(tmp_path, "start", sid).returncode == 0, sid
        assert ctl(tmp_path, "done", sid).returncode == 0, sid
    r = ctl(tmp_path, "start", "s12")
    assert r.returncode != 0 and "OK" in (r.stdout + r.stderr)
    assert ctl(tmp_path, "approve", "s12", "--by", "テスト").returncode == 0
    assert ctl(tmp_path, "start", "s12").returncode == 0


def test_fail_reset_and_next(tmp_path):
    ctl(tmp_path, "start", "s0")
    assert ctl(tmp_path, "fail", "s0", "--note", "欠け").returncode == 0
    assert "失敗" in ctl(tmp_path, "next").stdout
    assert ctl(tmp_path, "reset", "s0").returncode == 0
    assert state(tmp_path)["s0"]["status"] == "pending"
    assert "s0" in ctl(tmp_path, "next").stdout


def test_train_years_other_than_default_stops(tmp_path):
    ctl(tmp_path, "decide", "train_years", "2016-2025", "--confirm")
    out = ctl(tmp_path, "next").stdout
    assert "止まって" in out and "次: " not in out
