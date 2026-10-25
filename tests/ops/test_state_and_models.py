"""state_sync（入出力の同期と上限）と model_publish（検証つき配布）。MemoryIO で完結。"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.ops import model_publish as MP, state_sync as SS
from src.ops.gcs_io import MemoryIO


def _write(p: Path, data: bytes) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


# ── state_sync ──

def test_push_then_pull_roundtrip(tmp_path):
    src, dst, io = tmp_path / "a", tmp_path / "b", MemoryIO()
    _write(src / "data/out/x.json", b'{"a":1}')
    _write(src / "data/out/sub/y.bin", b"1234")
    r = SS.push(io, [{"path": "data/out", "max_mb": 1}], src)
    assert r["uploaded"] == 2 and io.exists("state/data/out/sub/y.bin")
    r2 = SS.pull(io, [{"path": "data/out", "max_mb": 1}], dst)
    assert r2["downloaded"] == 2 and (dst / "data/out/x.json").read_bytes() == b'{"a":1}'
    assert SS.pull(io, [{"path": "data/out", "max_mb": 1}], dst)["skipped_same"] == 2          # 取り直さない


def test_push_only_changed_since(tmp_path):
    io = MemoryIO()
    old = _write(tmp_path / "data/out/old.json", b"o")
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    _write(tmp_path / "data/out/new.json", b"n")
    SS.push(io, [{"path": "data/out", "max_mb": 1}], tmp_path, since=time.time() - 60)
    assert io.exists("state/data/out/new.json") and not io.exists("state/data/out/old.json")


def test_budget_exceeded_raises_without_downloading(tmp_path):
    io = MemoryIO()
    io.put_bytes("state/data/big/f.bin", b"x" * (2 * SS.MB))
    with pytest.raises(SS.StateBudgetError):
        SS.pull(io, [{"path": "data/big", "max_mb": 1}], tmp_path)
    assert not (tmp_path / "data/big/f.bin").exists()
    _write(tmp_path / "data/bigout/f.bin", b"x" * (2 * SS.MB))
    with pytest.raises(SS.StateBudgetError):
        SS.push(io, [{"path": "data/bigout", "max_mb": 1}], tmp_path)


def test_path_templates_and_traversal_guard(tmp_path):
    assert SS.expand([{"path": "data/{year}/x", "max_mb": 5}], {"year": "2026"}) == [("data/2026/x", 5)]
    with pytest.raises(ValueError):
        SS.expand([{"path": "../etc", "max_mb": 1}])
    with pytest.raises(ValueError):
        SS.expand([{"path": "/abs", "max_mb": 1}])


def test_plan_reports_local_sizes(tmp_path):
    _write(tmp_path / "data/in/a.bin", b"x" * 1024)
    rows = SS.plan([{"path": "data/in", "max_mb": 1}, {"path": "data/none", "max_mb": 1}], tmp_path)
    assert rows[0]["exists"] and rows[0]["files"] == 1 and not rows[0]["over"] and not rows[1]["exists"]


def test_pull_prefix_does_not_match_sibling_dirs(tmp_path):
    io = MemoryIO()
    io.put_bytes("state/data/in/a", b"1")
    io.put_bytes("state/data/in2/b", b"2")
    SS.pull(io, [{"path": "data/in", "max_mb": 1}], tmp_path)
    assert (tmp_path / "data/in/a").exists() and not (tmp_path / "data/in2/b").exists()


# ── model_publish ──

def _model(tmp_path: Path) -> Path:
    d = tmp_path / "bundle"
    _write(d / "meta.json", b'{"v":1}')
    _write(d / "m/model.txt", b"weights" * 100)
    return d


def test_publish_fetch_roundtrip_and_idempotent(tmp_path):
    io = MemoryIO()
    now = datetime(2026, 10, 21, tzinfo=timezone.utc)
    r = MP.publish(io, "bundle_v5", _model(tmp_path), now=now)
    assert r["uploaded_files"] == 2 and r["version"].startswith("20261021-")
    assert MP.publish(io, "bundle_v5", _model(tmp_path), now=now)["uploaded_files"] == 0        # 同じ内容は再アップロードしない
    got = MP.fetch(io, "bundle_v5", tmp_path / "cache")
    assert (got / "m/model.txt").read_bytes() == b"weights" * 100
    assert MP.fetch(io, "bundle_v5", tmp_path / "cache") == got                                   # 検証済みなら取り直さない


def test_fetch_only_prefix_takes_just_those_files(tmp_path):
    io = MemoryIO()
    MP.publish(io, "bundle_v5", _model(tmp_path))
    got = MP.fetch(io, "bundle_v5", tmp_path / "cache", only_prefix="m/")
    assert (got / "m/model.txt").is_file() and not (got / "meta.json").exists()
    assert MP.fetch(io, "bundle_v5", tmp_path / "cache", only_prefix="m/") == got                 # 検証済みなら取り直さない
    with pytest.raises(MP.ModelIntegrityError):
        MP.fetch(io, "bundle_v5", tmp_path / "cache", only_prefix="nothing/")


def test_latest_is_written_last_and_points_to_manifest(tmp_path):
    io = MemoryIO()
    MP.publish(io, "c", _model(tmp_path))
    latest = json.loads(io.get_bytes("models/c/latest.json"))
    assert io.exists(f"models/c/{latest['version']}/manifest.json")


def test_fetch_rejects_corruption_and_keeps_old(tmp_path):
    io = MemoryIO()
    v = MP.publish(io, "c", _model(tmp_path))["version"]
    good = MP.fetch(io, "c", tmp_path / "cache")
    io.put_bytes(f"models/c/{v}/m/model.txt", b"tampered")
    (good / ".verified.json").unlink()                      # 検証済みの印を消して再検証させる
    with pytest.raises(MP.ModelIntegrityError):
        MP.fetch(io, "c", tmp_path / "cache")
    assert not [p for p in (tmp_path / "cache").iterdir() if p.name.startswith(".c.")]           # 一時ディレクトリは残さない


def test_manifest_hash_mismatch_with_latest_is_rejected(tmp_path):
    io = MemoryIO()
    MP.publish(io, "c", _model(tmp_path))
    latest = json.loads(io.get_bytes("models/c/latest.json"))
    io.put_bytes(f"models/c/{latest['version']}/manifest.json", b'{"version":"x","files":[],"total_bytes":0}')
    with pytest.raises(MP.ModelIntegrityError):
        MP.resolve_version(io, "c")


def test_fetch_size_limit_and_missing_latest(tmp_path):
    io = MemoryIO()
    with pytest.raises(MP.ModelIntegrityError):
        MP.fetch(io, "none", tmp_path)
    big = tmp_path / "big"
    _write(big / "f", b"x" * (2 * MP.MB))
    MP.publish(io, "big", big)
    with pytest.raises(MP.ModelIntegrityError):
        MP.fetch(io, "big", tmp_path / "cache", max_mb=1)


def test_single_file_component_and_install_to(tmp_path):
    io = MemoryIO()
    f = _write(tmp_path / "models/final_odds_bundle.json", b'{"heads":[]}')
    MP.publish(io, "final_odds", f)
    models = {"final_odds": {"local_path": "models/final_odds_bundle.json", "entry": "final_odds_bundle.json", "install_to": "models/final_odds_bundle.json"},
              "tl": {"env": "KEIBA_V5_TL_REF", "local_path": "x/tl.parquet", "entry": "tl.parquet"}, "d": {"env": "KEIBA_V5_BUNDLE", "local_path": "x/d"}}
    got = MP.fetch(io, "final_odds", tmp_path / "cache")
    root = tmp_path / "app"
    link = MP.install(models, "final_odds", got, root)
    assert link == root / "models/final_odds_bundle.json" and link.read_bytes() == b'{"heads":[]}'
    assert MP.install(models, "final_odds", got, root) == link                    # 繰り返しても壊れない
    assert MP.env_for(models, "tl", Path("/c/tl/1")) == ("KEIBA_V5_TL_REF", "/c/tl/1/tl.parquet")
    assert MP.env_for(models, "d", Path("/c/d/1")) == ("KEIBA_V5_BUNDLE", "/c/d/1")
    assert MP.install(models, "d", Path("/c/d/1"), root) is None


def test_publish_empty_dir_fails(tmp_path):
    (tmp_path / "e").mkdir()
    with pytest.raises(FileNotFoundError):
        MP.publish(MemoryIO(), "e", tmp_path / "e")


def test_job_map_model_components_are_wellformed():
    from src.ops import job_map

    for name, spec in job_map.load()["models"].items():
        assert spec.get("local_path"), name
        assert spec.get("env") or spec.get("install_to"), f"{name}: env か install_to のどちらかが必要"


def test_snapshot_pushes_only_existing_parity_paths(tmp_path):
    from src.ops import job_map

    io = MemoryIO()
    _write(tmp_path / "data/local/features/jockey_trainer_stats/x.parquet", b"abc")
    res = SS.snapshot(io, job_map.load(), tmp_path, "20261024")
    by = {r["job"]: r for r in res}
    assert by["jockey-trainer-stats"]["uploaded"] == 1 and io.exists("host_snapshot/20261024/data/local/features/jockey_trainer_stats/x.parquet")
    assert "ホストに成果物が無い" in by["export-tables-daily"]["skipped"]
    assert "gcp-daily-cost-report" not in by                                   # parity 対象外は載らない


def test_host_snapshot_cron_installer_is_valid_shell():
    import subprocess

    assert subprocess.run(["bash", "-n", "scripts/cron/install_host_snapshot_cron.sh"]).returncode == 0


def test_state_include_limits_files_in_plan_pull_and_push(tmp_path):
    root = tmp_path / "r"
    for y in ("2025", "2026"):
        _write(root / f"data/t/{y}/race_result_flat.parquet", b"a" * 100)
        _write(root / f"data/t/{y}/smartrc_race_race.parquet", b"b" * 5000)
    entries = [{"path": "data/t", "max_mb": 1, "include": ["*/race_result_flat.parquet"]}]
    assert SS.plan(entries, root)[0]["files"] == 2 and SS.plan([{"path": "data/t", "max_mb": 1}], root)[0]["files"] == 4
    io = MemoryIO()
    SS.push(io, [{"path": "data/t", "max_mb": 1}], root)                       # 先に全部上げておく
    dest = tmp_path / "d"
    got = SS.pull(io, entries, dest)
    assert got["downloaded"] == 2 and (dest / "data/t/2026/race_result_flat.parquet").exists() and not (dest / "data/t/2026/smartrc_race_race.parquet").exists()
    dest2 = tmp_path / "d2"
    io2 = MemoryIO()
    SS.push(io2, entries, root)
    assert sorted(io2.list("state/")) == ["state/data/t/2025/race_result_flat.parquet", "state/data/t/2026/race_result_flat.parquet"]
