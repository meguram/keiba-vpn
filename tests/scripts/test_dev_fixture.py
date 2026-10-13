"""dev_fixture: git 配布する最小データ（実データ 1 開催日）のパック/展開。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.scripts.data import dev_fixture as F


@pytest.fixture()
def env(tmp_path, monkeypatch):
    mock, ref, fx = tmp_path / "mock", tmp_path / "ref", tmp_path / "fx"
    monkeypatch.setenv("KEIBA_DEV_MOCK_DIR", str(mock))
    monkeypatch.setenv("KEIBA_PAGE_REFERENCE_DIR", str(ref))
    monkeypatch.setattr(F, "FIXTURE_DIR", fx)
    monkeypatch.setattr(F, "ARCHIVE", fx / "real_day.tar.gz")
    monkeypatch.setattr(F, "MANIFEST", fx / "MANIFEST.json")
    return mock, ref


def _w(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _seed(mock: Path, ref: Path, date="20260906", rids=("202601020601", "202601020602")):
    _w(ref / "race_lists" / f"{date}.json", {"date": date, "races": [{"race_id": r} for r in rids]})
    _w(ref / "race_day_schedule" / f"{date}.json", {"date_fmt": date, "slots": []})
    for r in rids:
        for cat in ("race_shutuba", "race_result", "race_odds", "race_pair_odds", "race_predictions", "final_odds_prediction"):
            _w(mock / cat / r[:4] / f"{r}.json", {"race_id": r, "cat": cat, "名前": "馬"})


def test_pack_is_reproducible_and_manifest_matches(env):
    mock, ref = env
    _seed(mock, ref)
    m1 = F.pack("20260906")
    h1 = hashlib.sha256(F.ARCHIVE.read_bytes()).hexdigest()
    F.pack("20260906")
    assert hashlib.sha256(F.ARCHIVE.read_bytes()).hexdigest() == h1 == m1["archive_sha256"]
    assert m1["races"] == 2 and m1["counts"]["race_shutuba"] == 2 and m1["counts"]["race_lists"] == 1


def test_pack_refuses_incomplete_day(env):
    mock, ref = env
    _seed(mock, ref)
    (mock / "race_odds" / "2026" / "202601020602.json").unlink()
    with pytest.raises(SystemExit):
        F.pack("20260906")


def test_install_restores_into_empty_store_and_is_idempotent(env, tmp_path, monkeypatch):
    mock, ref = env
    _seed(mock, ref)
    F.pack("20260906")
    # 別 PC を模擬: ストアを空の別ディレクトリにする
    monkeypatch.setenv("KEIBA_DEV_MOCK_DIR", str(tmp_path / "pc2_mock"))
    monkeypatch.setenv("KEIBA_PAGE_REFERENCE_DIR", str(tmp_path / "pc2_ref"))
    assert F.status()["real_day_installed_races"] == 0
    assert F.install() == {"written": 14, "kept": 0}
    assert F.status()["real_day_installed_races"] == 2
    assert json.loads((tmp_path / "pc2_mock/race_shutuba/2026/202601020601.json").read_text())["名前"] == "馬"
    # 2 回目: 既存の実データは上書きしない
    assert F.install() == {"written": 0, "kept": 14}


def test_install_overwrites_mock_flagged_files_but_keeps_real_ones(env, tmp_path, monkeypatch):
    mock, ref = env
    _seed(mock, ref)
    F.pack("20260906")
    monkeypatch.setenv("KEIBA_DEV_MOCK_DIR", str(tmp_path / "m2"))
    monkeypatch.setenv("KEIBA_PAGE_REFERENCE_DIR", str(tmp_path / "r2"))
    _w(tmp_path / "m2/race_odds/2026/202601020601.json", {"_meta": {"dev_mock": True}, "x": 1})  # モック → 上書き
    _w(tmp_path / "m2/race_shutuba/2026/202601020601.json", {"keep": "me"})  # 実データ相当 → 維持
    r = F.install()
    assert r["kept"] == 1 and r["written"] == 13
    assert json.loads((tmp_path / "m2/race_odds/2026/202601020601.json").read_text())["cat"] == "race_odds"
    assert json.loads((tmp_path / "m2/race_shutuba/2026/202601020601.json").read_text()) == {"keep": "me"}
    assert F.install(force=True)["kept"] == 0


def test_install_rejects_path_traversal(env, tmp_path):
    import io
    import tarfile

    F.FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(F.ARCHIVE, "w:gz") as tf:
        ti = tarfile.TarInfo("dev_mock/../../evil.json")
        ti.size = 2
        tf.addfile(ti, io.BytesIO(b"{}"))
    with pytest.raises(SystemExit):
        F.install()


def test_committed_fixture_is_consistent():
    """リポジトリに入っている dev_fixture/ が MANIFEST どおりで、36 レース分が展開できる。"""
    repo_fixture = F.REPO_ROOT / "dev_fixture"
    man = json.loads((repo_fixture / "MANIFEST.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((repo_fixture / man["archive"]).read_bytes()).hexdigest() == man["archive_sha256"]
    assert man["races"] == 36 and all(man["counts"][c] == 36 for c in ("race_shutuba", "race_result", "race_odds", "race_pair_odds", "race_predictions"))
