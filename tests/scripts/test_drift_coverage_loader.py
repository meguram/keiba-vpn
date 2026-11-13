import json

import pytest

from src.scripts.data import drift_coverage as C


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    import src.utils.project_env as PE

    monkeypatch.setattr(PE, "load_project_dotenv", lambda *a, **k: None)   # .env の値を os.environ に残さない


def _local(tmp_path, hid="2024100001"):
    p = tmp_path / "data/local/horse_training/2024" / f"{hid}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"horse_id": hid, "entries": []}), encoding="utf-8")
    return p


def test_default_loader_reads_the_local_copy_and_never_gcs_outside_gcp(tmp_path, monkeypatch):
    """ホスト（GCP の外）では GCS を読まない。手元に無い馬は None（2026-11-13: 1 頭ずつの GCS 読みで約 0.6GB の転送料）。"""
    _local(tmp_path)
    import src.scraper.storage as S

    for k in ("CLOUD_RUN_JOB", "K_SERVICE", "KEIBA_GCP_FORBIDDEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GCS_BUCKET", "b")

    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("GCP の外で GCS を読もうとした")

    monkeypatch.setattr(S, "HybridStorage", Boom)
    out = C.default_loader(root=str(tmp_path))(["2024100001", "2024100002"])
    assert out["2024100001"]["horse_id"] == "2024100001" and out["2024100002"] is None


def test_default_loader_falls_back_to_storage_inside_gcp(tmp_path, monkeypatch):
    """Cloud Run Jobs（手元の複製が無い）では、既定のままで GCS に戻る。"""
    import src.scraper.storage as S

    class Fake:
        def load(self, kind, h):
            return {"horse_id": h, "from": "gcs"}

    monkeypatch.setattr(S, "HybridStorage", Fake)
    monkeypatch.setenv("GCS_BUCKET", "b")
    monkeypatch.setenv("CLOUD_RUN_JOB", "keiba-dispatch-s")
    out = C.default_loader(root=str(tmp_path))(["2024100009"])
    assert out["2024100009"]["from"] == "gcs"


def test_broken_local_file_is_logged_not_silent(tmp_path, monkeypatch, caplog):
    p = _local(tmp_path)
    p.write_text("{壊れた", encoding="utf-8")
    out = C.default_loader(root=str(tmp_path), allow_gcs=False)(["2024100001"])
    assert out["2024100001"] is None and "読めない" in caplog.text
