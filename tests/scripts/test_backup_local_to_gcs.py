"""ローカルデータの GCS バックアップ(backup_local_to_gcs)の取り決めのテスト。実際の GCS には接続しない。"""

from __future__ import annotations

import sqlite3
import subprocess

from src.scripts.maintenance import backup_local_to_gcs as B


def test_unit_names_are_unique_and_paths_are_relative():
    names = [u.name for u in B.UNITS]
    assert len(names) == len(set(names))
    for u in B.UNITS:
        for p in u.paths + u.excludes:
            assert not p.startswith("/") and ".." not in p.split("/"), p


def test_secrets_are_excluded_from_every_unit(tmp_path):
    for u in B.UNITS:
        cmd = B.tar_command(u, tmp_path, sudo=False)
        for pat in ("*cookie*", "*credential*", "*secret*", "*.pem", "*.key", "*/.env"):
            assert pat in cmd, (u.name, pat)


def test_tar_command_skips_missing_paths_and_excludes(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "keep.txt").write_text("x")
    (tmp_path / "a" / "cookie_jar.txt").write_text("secret")
    (tmp_path / "a" / "sub").mkdir()
    (tmp_path / "a" / "sub" / "drop.txt").write_text("y")
    unit = B.Unit("t", ["a", "nope"], excludes=["a/sub"])
    cmd = B.tar_command(unit, tmp_path, sudo=False)
    assert "nope" not in cmd and "a" in cmd
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    names = subprocess.run(["tar", "-t"], input=out, capture_output=True, check=True).stdout.decode().split()
    assert "a/keep.txt" in names and "a/cookie_jar.txt" not in names and "a/sub/drop.txt" not in names


def test_backup_never_lands_under_the_scraped_data_prefix():
    assert B.GCS_BACKUP_ROOT.startswith("chuou/backup/") and "preprocessed" not in B.GCS_BACKUP_ROOT


def test_sqlite_backup_api_gives_a_consistent_copy(tmp_path):
    src = tmp_path / "s.db"
    c = sqlite3.connect(src); c.execute("create table t(a)"); c.executemany("insert into t values (?)", [(i,) for i in range(5)]); c.commit(); c.close()
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True); d = sqlite3.connect(tmp_path / "d.db")
    with d:
        s.backup(d)
    assert d.execute("select count(*) from t").fetchone()[0] == 5
