"""day-prep（src/ops/jobs/day_prep.py）: 日単位キャッシュだけを作り、予測は作らない。重い本体は fake で置き換える。"""
import types

import pytest

from src.ops.jobs import day_prep as DP
import src.pipeline.inference.race_day_workflow as wf


def test_no_races_is_skipped(monkeypatch):
    monkeypatch.setattr(wf, "load_day_schedule", lambda st, d: [])
    assert DP.run("20261024", storage=object(), ctx=object())["status"] == "skipped"


def test_requires_v6(monkeypatch):
    monkeypatch.setattr(wf, "load_day_schedule", lambda st, d: [{"race_id": "202605040101"}])
    with pytest.raises(RuntimeError, match="v6 が有効でない"):
        DP.run("20261024", storage=object(), ctx=types.SimpleNamespace(v6=None))


def test_cli_rejects_bad_date_and_reports_failure(monkeypatch):
    assert DP.main(["--date", "2026"]) == 2
    monkeypatch.setattr(DP, "run", lambda *a, **k: {"status": "error"})
    assert DP.main(["--date", "20261024"]) == 1
    monkeypatch.setattr(DP, "run", lambda *a, **k: {"status": "ok"})
    assert DP.main(["--date", "20261024"]) == 0
