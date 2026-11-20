"""起点年の記録と照合（B7、src/pipeline/models/training_years.py）。env の有無で起点年が変わり、束と食い違えば止まる。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run(env_fy: str | None, code: str) -> str:
    env = {k: v for k, v in os.environ.items() if k != "KEIBA_FIRST_YEAR"}
    if env_fy is not None:
        env["KEIBA_FIRST_YEAR"] = env_fy
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


@pytest.mark.parametrize("env_fy,want,src", [(None, 2020, "default"), ("2016", 2016, "env"), ("2020", 2020, "env")])
def test_stamp_follows_env(env_fy, want, src):
    out = json.loads(_run(env_fy, "import json; from src.pipeline.models import training_years as t; print(json.dumps(t.first_year_stamp(2016)))"))
    assert out["first_year"] == want and out["first_year_source"] == src
    assert out["training_years"] == [str(want), "2025"] and out["train_year_min"] == 2016


def test_check_first_year(tmp_path):
    from src.pipeline.models import training_years as t

    assert t.check_first_year(t.FIRST_YEAR, where="x") is None
    with pytest.raises(SystemExit, match="起点年"):
        t.check_first_year(t.FIRST_YEAR - 4, where="学習の道具")
    assert "記録は" in t.check_first_year(t.FIRST_YEAR - 4, where="推論", strict=False)
    assert "古い束" in t.check_first_year(None, where="推論")             # 記録の無い古い束は止めない
    (tmp_path / "bundle.json").write_text(json.dumps({"members": []}), encoding="utf-8")
    assert t.bundle_first_year(tmp_path) is None
    (tmp_path / "fit_info.json").write_text(json.dumps({"first_year": 2016}), encoding="utf-8")
    assert t.bundle_first_year(tmp_path) == 2016


def test_bundle_save_writes_first_year(tmp_path):
    from src.pipeline.models import training_years as t
    from src.pipeline.models.ensemble_bundle import EnsembleBundle

    b = EnsembleBundle([], {"targets": {}}, {})
    b.save(tmp_path)
    meta = json.loads((tmp_path / "bundle.json").read_text(encoding="utf-8"))
    assert meta["first_year"] == t.FIRST_YEAR and t.bundle_first_year(tmp_path) == t.FIRST_YEAR
