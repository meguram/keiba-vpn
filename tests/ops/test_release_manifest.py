"""リリース・マニフェストと stg/prod 照合。外部接続なし。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.ops import release_manifest as RM
from src.scripts.ops import env_parity_check as EP


def _root(tmp_path: Path, table: dict | None = None) -> Path:
    (tmp_path / "config").mkdir()
    (tmp_path / "config/prob_calibration.json").write_text('{"a": 1}')
    (tmp_path / "config/maintenance.json").write_text('{"x": 1}')
    (tmp_path / "src/api").mkdir(parents=True)
    (tmp_path / "src/api/m.py").write_text("x = 1\n")
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend/a.tsx").write_text("export {}")
    if table is not None:
        (tmp_path / "config/behavior_flags.json").write_text(json.dumps(table))
    return tmp_path


def _m(root: Path, env: dict, **kw):
    RM._CACHE.clear()
    return RM.build_manifest(root=root, env=env, **kw)


def test_secrets_and_paths_are_not_exposed(tmp_path):
    root = _root(tmp_path)
    m = _m(root, {"KEIBA_ENV": "stg", "KEIBA_MODEL_STORE": "gs://secret-bucket/x", "KEIBA_V3_BUNDLE": "/a/b/bundle_x",
                  "SLACK_WEBHOOK_URL": "https://hooks.example/XXX", "KEIBA_PROB_CALIBRATION": "1", "GCS_BUCKET": "bk"})
    text = json.dumps(m)
    assert "secret-bucket" not in text and "hooks.example" not in text and "/a/b" not in text and "GCS_BUCKET" not in text
    assert m["flags"]["KEIBA_V3_BUNDLE"] == "bundle_x" and m["flags"]["KEIBA_MODEL_STORE"].startswith("sha:")
    assert m["flags"]["KEIBA_PROB_CALIBRATION"] == "1" and m["env"] == "stg"
    assert "config/maintenance.json" not in m["configs"]["files"]


def test_config_and_code_change_the_digest(tmp_path):
    root = _root(tmp_path)
    a = _m(root, {})
    (root / "config/prob_calibration.json").write_text('{"a": 2}')
    (root / "src/api/m.py").write_text("x = 2\n")
    b = _m(root, {})
    paths = {d["path"] for d in RM.blocking(RM.compare(a, b))}
    assert "configs.digest" in paths and "code.tree" in paths and "code.dirs.src/api" in paths


def test_intentional_differences_are_ignored(tmp_path):
    root = _root(tmp_path)
    a = _m(root, {"KEIBA_ENV": "stg", "KEIBA_DEPLOYMENT_LABEL": "STG"})
    b = _m(root, {"KEIBA_ENV": "prod"})
    assert RM.blocking(RM.compare(a, b)) == []


def test_flag_difference_blocks_unless_declared(tmp_path):
    root = _root(tmp_path)
    a = _m(root, {"KEIBA_ENV": "stg", "KEIBA_PACE_TYPE": "1"})
    b = _m(root, {"KEIBA_ENV": "prod"})
    assert [d["path"] for d in RM.blocking(RM.compare(a, b))] == ["flags.KEIBA_PACE_TYPE"]
    assert RM.blocking(RM.compare(a, b, allow={"flags.KEIBA_PACE_TYPE"})) == []


def test_flag_table_supplies_defaults_and_detects_drift(tmp_path):
    root = _root(tmp_path, {"defaults": {"KEIBA_PROB_CALIBRATION": "1"}, "env_overrides": {"stg": {}, "prod": {}}, "env_overridable": []})
    m = _m(root, {"KEIBA_ENV": "prod"})
    assert m["flags"]["KEIBA_PROB_CALIBRATION"] == "1" and m["flag_sources"]["KEIBA_PROB_CALIBRATION"] == "file" and m["flag_drift"] == []
    m2 = _m(root, {"KEIBA_ENV": "prod", "KEIBA_PROB_CALIBRATION": "0"})
    assert m2["flag_drift"] == ["KEIBA_PROB_CALIBRATION"]
    (tmp_path / "r2").mkdir()
    root2 = _root(tmp_path / "r2",
                  {"defaults": {"KEIBA_PROB_CALIBRATION": "1"}, "env_overridable": ["KEIBA_PROB_CALIBRATION"]})
    assert _m(root2, {"KEIBA_ENV": "prod", "KEIBA_PROB_CALIBRATION": "0"})["flag_drift"] == []


def test_apply_behavior_flags_does_not_override_explicit_env(tmp_path):
    from src.utils.behavior_flags import apply_behavior_flags

    root = _root(tmp_path, {"defaults": {"KEIBA_PACE_TYPE": "1", "KEIBA_LAP_FORECAST": "1"}, "env_overrides": {"stg": {"KEIBA_LAP_FORECAST": "0"}}})
    env = {"KEIBA_ENV": "stg", "KEIBA_PACE_TYPE": "0"}
    applied = apply_behavior_flags(root, env)
    assert env["KEIBA_PACE_TYPE"] == "0" and env["KEIBA_LAP_FORECAST"] == "0" and applied == {"KEIBA_LAP_FORECAST": "0"}
    assert apply_behavior_flags(tmp_path / "none", {}) == {}


def test_bundle_fingerprint_uses_manifest_then_name_size(tmp_path):
    b = tmp_path / "bundle"
    b.mkdir()
    (b / "model.bin").write_bytes(b"abc")
    f1 = RM.bundle_fingerprint(b)
    assert f1["basis"] == "name+size"
    (b / "manifest.json").write_text(json.dumps({"version": "v5", "n_features": 3, "files": {"model.bin": {"sha256": "aa"}}}))
    f2 = RM.bundle_fingerprint(b)
    assert f2["basis"] == "manifest" and f2["version"] == "v5" and f2["digest"] != f1["digest"]
    assert RM.bundle_fingerprint(tmp_path / "nope")["missing"] is True


def test_frontend_digest_ignores_build_artifacts(tmp_path):
    fe = tmp_path / "frontend"
    (fe / "node_modules/x").mkdir(parents=True)
    (fe / "a.tsx").write_text("1")
    d1 = RM.frontend_digest(fe)
    (fe / "node_modules/x/i.js").write_text("zzz")
    (fe / ".next").mkdir()
    (fe / ".next/c.js").write_text("zzz")
    (fe / "public").mkdir()
    (fe / "public/version.json").write_text("{}")
    assert RM.frontend_digest(fe) == d1
    (fe / "a.tsx").write_text("2")
    assert RM.frontend_digest(fe) != d1


def test_write_frontend_version_keeps_existing_keys(tmp_path):
    fe = tmp_path / "frontend"
    fe.mkdir()
    (fe / "a.ts").write_text("1")
    out = fe / "public/version.json"
    out.parent.mkdir()
    out.write_text(json.dumps({"release": "r1", "commit": "abc"}))
    v = RM.write_frontend_version(fe, "prod", out)
    assert v["release"] == "r1" and v["env"] == "prod" and v["source_digest"] == RM.frontend_digest(fe)


def test_parity_compare_files(tmp_path, capsys):
    root = _root(tmp_path)
    stg = _m(root, {"KEIBA_ENV": "stg"})
    prod = _m(root, {"KEIBA_ENV": "prod"})
    (tmp_path / "s.json").write_text(json.dumps(stg))
    (tmp_path / "p.json").write_text(json.dumps(prod))
    assert EP.main(["compare", "--stg-file", str(tmp_path / "s.json"), "--prod-file", str(tmp_path / "p.json")]) == 0
    prod["flags"]["KEIBA_PACE_TYPE"] = "1"
    (tmp_path / "p.json").write_text(json.dumps(prod))
    assert EP.main(["compare", "--stg-file", str(tmp_path / "s.json"), "--prod-file", str(tmp_path / "p.json")]) == 1
    assert "KEIBA_PACE_TYPE" in capsys.readouterr().out


def test_parity_flags_drift_is_blocking(tmp_path):
    root = _root(tmp_path, {"defaults": {"KEIBA_PACE_TYPE": "1"}})
    stg = _m(root, {"KEIBA_ENV": "stg", "KEIBA_PACE_TYPE": "0"})
    prod = _m(root, {"KEIBA_ENV": "prod", "KEIBA_PACE_TYPE": "1"})
    bad, _ = EP.summarize(RM.compare(stg, prod), (stg, prod))
    assert any(d["path"] == "stg.flag_drift" for d in bad)


def test_parity_unreachable_is_exit_2(tmp_path):
    assert EP.main(["compare", "--stg-file", str(tmp_path / "missing.json"), "--prod-file", str(tmp_path / "missing.json")]) == 2


def test_manifest_endpoint_requires_token(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.release_manifest_api import router

    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)
    monkeypatch.delenv("KEIBA_MANIFEST_TOKEN", raising=False)
    assert c.get("/api/ops/release-manifest").status_code == 404
    monkeypatch.setenv("KEIBA_MANIFEST_TOKEN", "tok123")
    assert c.get("/api/ops/release-manifest").status_code == 404
    assert c.get("/api/ops/release-manifest", headers={"X-Keiba-Manifest-Token": "bad"}).status_code == 404
    r = c.get("/api/ops/release-manifest", headers={"X-Keiba-Manifest-Token": "tok123"})
    assert r.status_code == 200 and r.json()["schema"] == 1 and "tok123" not in r.text
