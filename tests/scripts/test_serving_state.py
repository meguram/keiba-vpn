"""serving_state の push/pull（DirBackend で GCS の代わり）。外部接続なし。"""

import io
import json
import tarfile
from pathlib import Path

import pytest

from src.scripts.ops import serving_state as SS

CFG = {
    "object_prefix": "serving_state",
    "default_groups": ["core"],
    "groups": {"core": {"include": ["data/page_reference/race_lists/**", "data/page_reference/tables/*/megu_index_flat.parquet",
                                    "data/local/inference/lap_forecast/**"],
                        "exclude": ["*.tmp", "*.lock"]}},
    "limits": {"max_total_bytes": 10_000, "max_files": 100},
}


def make_tree(root: Path):
    files = {
        "data/page_reference/race_lists/20261018.json": b'{"a":1}',
        "data/page_reference/tables/2025/megu_index_flat.parquet": b"PAR1" * 10,
        "data/page_reference/tables/2025/race_odds_flat.parquet": b"x" * 5000,        # 入らない（3GB 側）
        "data/local/inference/lap_forecast/2026/x.json": b"{}",
        "data/local/inference/lap_forecast/2026/x.tmp": b"tmp",                         # exclude
    }
    for rel, b in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)


def test_select_only_wanted_files(tmp_path):
    make_tree(tmp_path)
    rels = [p.relative_to(tmp_path).as_posix() for p in SS.select_files(tmp_path, CFG["groups"]["core"])]
    assert rels == ["data/local/inference/lap_forecast/2026/x.json", "data/page_reference/race_lists/20261018.json",
                    "data/page_reference/tables/2025/megu_index_flat.parquet"]


def test_push_pull_roundtrip_and_unchanged(tmp_path):
    src, dst, bucket = tmp_path / "src", tmp_path / "dst", tmp_path / "bucket"
    make_tree(src)
    be = SS.DirBackend(bucket)
    info = SS.push(be, CFG, "core", src)
    assert info["files"] == 3 and (bucket / "serving_state/core/latest.json").exists()
    r = SS.pull(be, CFG, "core", dst)
    assert r["status"] == "pulled" and r["files"] == 3
    assert (dst / "data/page_reference/race_lists/20261018.json").read_bytes() == b'{"a":1}'
    assert not (dst / "data/page_reference/tables/2025/race_odds_flat.parquet").exists()
    assert SS.pull(be, CFG, "core", dst)["status"] == "unchanged"                # 同じ sha は取り直さない


def test_pull_detects_sha_mismatch_and_missing(tmp_path):
    src, dst, bucket = tmp_path / "src", tmp_path / "dst", tmp_path / "bucket"
    make_tree(src)
    be = SS.DirBackend(bucket)
    assert SS.pull(be, CFG, "core", dst)["status"] == "no_state"
    info = SS.push(be, CFG, "core", src)
    obj = bucket / info["object"]
    obj.write_bytes(obj.read_bytes() + b"tamper")
    assert SS.pull(be, CFG, "core", dst)["status"] == "sha_mismatch"
    assert not (dst / "data").exists()                                           # 検証前には何も展開しない
    obj.unlink()
    assert SS.pull(be, CFG, "core", dst)["status"] == "missing_object"


def test_pull_refuses_unsafe_paths(tmp_path):
    bucket, dst = tmp_path / "bucket", tmp_path / "dst"
    (bucket / "serving_state/core").mkdir(parents=True)
    import hashlib

    for evil in ("../escape.txt", "/abs.txt", "etc/passwd", "data/../../x"):
        arc = bucket / "serving_state/core/e.tar.gz"
        with tarfile.open(arc, "w:gz") as t:
            ti = tarfile.TarInfo(evil)
            ti.size = 1
            t.addfile(ti, io.BytesIO(b"x"))
        sha = hashlib.sha256(arc.read_bytes()).hexdigest()
        (bucket / "serving_state/core/latest.json").write_text(json.dumps({"object": "serving_state/core/e.tar.gz", "sha256": sha}))
        with pytest.raises(ValueError):
            SS.pull(SS.DirBackend(bucket), CFG, "core", dst)
    assert not (tmp_path / "escape.txt").exists()


def test_limits_are_enforced(tmp_path):
    make_tree(tmp_path)
    big = {**CFG, "limits": {"max_total_bytes": 5, "max_files": 100}}
    with pytest.raises(SystemExit):
        SS.push(SS.DirBackend(tmp_path / "b"), big, "core", tmp_path)


def test_same_content_gives_same_sha(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_tree(a)
    make_tree(b)
    fa, fb = SS.select_files(a, CFG["groups"]["core"]), SS.select_files(b, CFG["groups"]["core"])
    assert SS.build_archive(a, fa, tmp_path / "1.tgz") == SS.build_archive(b, fb, tmp_path / "2.tgz")


def test_pull_default_groups_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_SERVING_STATE_LOCAL_DIR", str(tmp_path / "nothing"))
    res = SS.pull_default_groups()
    assert res and res[0]["status"] in ("no_state", "error")
    monkeypatch.delenv("KEIBA_SERVING_STATE_LOCAL_DIR")
    monkeypatch.setenv("KEIBA_ENV", "dev")
    assert SS.pull_default_groups()[0]["status"] == "backend_error"              # dev は GCS に触れない。落ちない


def test_repo_config_is_valid():
    cfg = SS.load_config()
    assert cfg["default_groups"] == ["core"] and set(cfg["groups"]) >= {"core", "live_compute"}
    for g in cfg["groups"].values():
        assert all(i.startswith("data/") for i in g["include"])                  # 展開は data/ 配下だけ
