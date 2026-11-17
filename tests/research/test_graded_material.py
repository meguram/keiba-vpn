"""材料の年 1 回の決め直し（src/research/graded/material.py）。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.research.graded import material as MT


def test_env_writes_outside_repo_and_sets_periods(tmp_path):
    e = MT.env_for(2025, tmp_path)
    assert e["GRADED_MATERIAL_END"] == e["GRADED_YEAR_END"] == e["FOLKLORE_YEAR_END"] == "2025"
    for k in ("GRADED_PREREG", "GRADED_PREREG_ROI", "GRADED_PREREG_KILL", "GRADED_OUT", "FOLKLORE_OUT"):
        assert e[k].startswith(str(tmp_path))
    assert "GRADED_RESELECT" not in e and "FOLKLORE_ENTRIES" not in e
    assert e["FOLKLORE_KEEP_DNF"] == "1" and e["FOLKLORE_RESUME"] == "0"
    r = MT.env_for(2025, tmp_path, resume=True)
    assert r["FOLKLORE_RESUME"] == "1" and r["GRADED_RESELECT"] == "1"


def test_periods_follow_material_end():
    code = ("from src.research.graded import run as R, tally as T; "
            "print('PERIODS', R.DISC, R.CONF, R.EVAL, R.DEV_PAY, T.FREEZE_END.date())")
    out = subprocess.run([sys.executable, "-c", code], env={**__import__("os").environ, "GRADED_MATERIAL_END": "2025"},
                         capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2], check=True).stdout
    line = next(x for x in out.splitlines() if x.startswith("PERIODS"))
    assert line == "PERIODS (2018, 2022) (2023, 2025) (2026, 2027) (2020, 2025) 2025-12-31"
    from src.research.graded import run as R
    from src.research.graded import tally as T
    assert (R.DISC, R.CONF, R.EVAL, R.DEV_PAY) == ((2018, 2021), (2022, 2024), (2025, 2026), (2020, 2024))   # 既定は最初の材料のまま
    assert str(T.FREEZE_END.date()) == "2024-12-31"
    P = json.loads((MT.REPO / "src/research/graded/prereg.json").read_text(encoding="utf-8"))
    assert [list(P["periods"][k]) for k in ("discovery", "confirm", "evaluate")] == [list(R.DISC), list(R.CONF), list(R.EVAL)]


def test_check_end():
    MT.check_end(2025, "2026-01-10")
    with pytest.raises(SystemExit):
        MT.check_end(2026, "2026-11-17")       # 今年はまだ結果がそろっていない
    with pytest.raises(SystemExit):
        MT.check_end(2023, "2026-11-17")


def test_refuses_out_inside_repo():
    with pytest.raises(SystemExit):
        MT.main(["--end", "2024", "--out", str(MT.REPO / "tmp_material")])
    assert not (MT.REPO / "tmp_material").exists()


def test_diff():
    old = {"periods": {"confirm": [2022, 2024]}, "sets": {"preday": {"used": ["A01", "B02"]}}}
    new = {"periods": {"confirm": [2023, 2025]}, "sets": {"preday": {"used": ["B02", "C03"]}}}
    d = MT.diff(old, new)
    assert d["preday_used_added"] == ["C03"] and d["preday_used_removed"] == ["A01"] and d["preday_used_kept"] == 1
    json.dumps(d)


def test_check_out_guards(tmp_path, monkeypatch):
    MT.check_out(tmp_path / "new", 2025, None)                                       # 新しい置き場は使える
    with pytest.raises(SystemExit):
        MT.check_out(tmp_path / "new", 2025, "select")                                # 始めていない置き場の続きは無い
    used = tmp_path / "used"
    used.mkdir()
    (used / "frame.parquet").write_bytes(b"x")
    with pytest.raises(SystemExit):
        MT.check_out(used, 2025, None)                                                # ほかの用途の置き場
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / MT.MARK).write_text('{"end": 2025}', encoding="utf-8")
    MT.check_out(mine, 2025, "select")
    with pytest.raises(SystemExit):
        MT.check_out(mine, 2026, "select")                                            # 区切りの年が違う
    with pytest.raises(SystemExit):
        MT.check_out(mine, 2025, None)                                                # 始めからやり直すのは新しい置き場で
    from src.research.graded import features as F
    with pytest.raises(SystemExit):
        MT.check_out(F.GRADED_OUT.resolve(), 2025, None)                             # 共有の作業場所


def test_diff_kill_roi():
    ok = {"versions": [{"name": "a", "primary": True, "used": ["X"]}]}
    nk = {"versions": [{"name": "b", "primary": True, "used": ["X", "Y"]}]}
    d = MT.diff({}, {}, ok, nk, None, None)
    assert d["kill"] == {"primary_old": "a", "primary_new": "b", "used_added": ["Y"], "used_removed": []}
    assert d["roi"]["primary_old"] is None
