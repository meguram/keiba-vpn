"""rb-state-daily（src/ops/jobs/rb_state.py）: 検査に通ったときだけ原子的に差し替え、通らなければ古い状態を残す。"""

import json
from pathlib import Path

import pandas as pd

from src.ops.jobs import rb_state as R


def _state(d: Path, n: int, cut: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": ["2026-10-01"] * n}).to_parquet(d / "hist.parquet", index=False)
    (d / "params.json").write_text(json.dumps({"cut": cut}), encoding="utf-8")


def _runner(n: int, cut: str):
    def run(mod: str, *args: str) -> None:
        if mod.endswith("build_rb_state"):
            _state(Path(args[args.index("--out") + 1]), n, cut)
        else:
            Path(args[args.index("--out") + 1]).mkdir(parents=True, exist_ok=True)
    return run


def test_first_build_then_swap(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "MIN_ROWS", 10)
    out = tmp_path / "state" / "rb_v6"
    res = R.run(out, [2026], runner=_runner(20, "2026-10-24"))
    assert res["status"] == "ok" and (out / "hist.parquet").exists()
    assert json.loads((out / "params.json").read_text())["cut"] == "2026-10-24"
    assert not [p for p in out.parent.iterdir() if p.name.startswith(".rb_v6_build.")]          # 作業用ディレクトリは残さない


def test_regression_keeps_old_state(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "MIN_ROWS", 10)
    out = tmp_path / "state" / "rb_v6"
    _state(out, 50, "2026-10-24")
    assert R.run(out, [2026], runner=_runner(20, "2026-10-25"))["status"] == "error"            # 件数が減った
    assert R.run(out, [2026], runner=_runner(60, "2026-10-01"))["status"] == "error"            # 最終日が戻った
    assert len(pd.read_parquet(out / "hist.parquet")) == 50 and json.loads((out / "params.json").read_text())["cut"] == "2026-10-24"
    assert R.run(out, [2026], runner=_runner(60, "2026-10-25"))["status"] == "ok"
    assert json.loads((out / "params.json").read_text())["cut"] == "2026-10-25"


def test_too_small_is_rejected(tmp_path):
    assert R.run(tmp_path / "s" / "rb_v6", [2026], runner=_runner(5, "2026-10-24"))["status"] == "error"
