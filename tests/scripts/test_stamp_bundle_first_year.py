"""stamp_bundle_first_year: 束の bundle.json に起点年を書き足す。別の値の上書きは止める。"""

import json

import pytest

from src.pipeline.models.training_years import bundle_first_year
from src.scripts.ops.stamp_bundle_first_year import stamp


def _bundle(tmp_path, **extra):
    (tmp_path / "bundle.json").write_text(json.dumps({"members": [], "spec": {}, **extra}), encoding="utf-8")
    return tmp_path


def test_stamp_adds_first_year_and_keeps_other_keys(tmp_path):
    d = _bundle(tmp_path)
    out = stamp(d, 2016, 2016)
    meta = json.loads((d / "bundle.json").read_text(encoding="utf-8"))
    assert out["changed"] and meta["first_year"] == 2016 and meta["train_year_min"] == 2016 and meta["members"] == []
    assert bundle_first_year(d) == 2016


def test_stamp_refuses_to_overwrite_a_different_year(tmp_path):
    d = _bundle(tmp_path, first_year=2020)
    with pytest.raises(SystemExit):
        stamp(d, 2016)
    assert json.loads((d / "bundle.json").read_text(encoding="utf-8"))["first_year"] == 2020


def test_dry_run_writes_nothing(tmp_path):
    d = _bundle(tmp_path)
    stamp(d, 2016, dry_run=True)
    assert bundle_first_year(d) is None


def test_stamp_refuses_conflict_with_fit_info(tmp_path):
    d = _bundle(tmp_path)
    (d / "fit_info.json").write_text(json.dumps({"first_year": 2020}), encoding="utf-8")
    with pytest.raises(SystemExit):
        stamp(d, 2016)


def test_stamp_keeps_existing_source_and_refuses_other_train_year_min(tmp_path):
    d = _bundle(tmp_path, first_year=2016, first_year_source="env", train_year_min=2016)
    assert stamp(d, 2016, 2016)["would_change"] is False
    assert json.loads((d / "bundle.json").read_text(encoding="utf-8"))["first_year_source"] == "env"
    with pytest.raises(SystemExit):
        stamp(d, 2016, 2017)


def test_dry_run_reports_would_change(tmp_path):
    d = _bundle(tmp_path)
    out = stamp(d, 2016, dry_run=True)
    assert out["would_change"] is True and out["changed"] is False and not (d / "bundle.json.tmp").exists()
