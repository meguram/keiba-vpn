"""KEIBA_ENV=dev: GCP 遮断とローカルモック（data/dev_mock）読み書きのテスト。"""

from __future__ import annotations

from datetime import date

import pytest

from src.config.gcp_guard import GcpAccessForbidden, assert_gcp_allowed, gcp_forbidden
from src.scraper.storage import HybridStorage
from src.scripts.data import make_dev_mock

TODAY = date(2026, 10, 6)  # 火曜 → 結果あり 2026-10-04 / 結果なし 2026-10-10


@pytest.fixture()
def dev_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "dev")
    monkeypatch.setenv("KEIBA_DEV_MOCK_DIR", str(tmp_path / "dev_mock"))
    monkeypatch.setenv("KEIBA_PAGE_REFERENCE_DIR", str(tmp_path / "page_reference"))
    monkeypatch.setenv("GCS_BUCKET", "some-prod-bucket")  # 設定されていても無視されること
    return tmp_path


@pytest.fixture()
def mock_storage(dev_env):
    make_dev_mock.generate(TODAY, make_dev_mock.dev_mock_root())
    return HybridStorage(base_dir=str(dev_env))


def test_guard_only_in_dev(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "dev")
    assert gcp_forbidden()
    with pytest.raises(GcpAccessForbidden):
        assert_gcp_allowed("x")
    for env in ("stg", "prod", ""):
        monkeypatch.setenv("KEIBA_ENV", env)
        assert not gcp_forbidden()
        assert_gcp_allowed("x")


def test_dev_storage_ignores_bucket_and_never_builds_client(dev_env):
    st = HybridStorage(base_dir=str(dev_env))
    assert st.gcs_enabled is False
    with pytest.raises(GcpAccessForbidden):
        st._get_bucket()


def test_credentials_and_cloud_clients_blocked(dev_env):
    from src.config.gcp_credentials import build_gcp_credentials
    from src.db.cloud_sql import get_cloud_sql_engine

    with pytest.raises(GcpAccessForbidden):
        build_gcp_credentials()
    with pytest.raises(GcpAccessForbidden):
        get_cloud_sql_engine("p:r:i", "u", "pw", "db")


def test_cloud_tasks_backend_disabled_in_dev(dev_env, monkeypatch):
    from src.scraper.cloud_tasks_queue import is_cloud_tasks_backend_enabled

    monkeypatch.setenv("KEIBA_QUEUE_BACKEND", "cloud_tasks")
    assert is_cloud_tasks_backend_enabled() is False


def test_mock_is_readable_through_hybrid_storage(mock_storage):
    st = mock_storage
    keys = st.list_keys("race_shutuba")
    assert len(keys) == 16
    assert st.list_keys("race_shutuba", "2026") == keys
    rid = keys[0]
    shutuba = st.load("race_shutuba", rid)
    assert len(shutuba["entries"]) == 8
    assert st.exists("race_shutuba", rid)
    assert not st.exists("race_shutuba", "000000000000")
    assert st.load("race_shutuba", "000000000000") is None
    assert set(st.batch_check_keys("race_shutuba", keys)) == set(keys)
    assert set(st.batch_list_blobs("race_shutuba", "2026")) == set(keys)

    horse = st.load("horse_result", shutuba["entries"][0]["horse_id"])
    assert horse["race_history"]
    ped = st.load("horse_pedigree_5gen", shutuba["entries"][0]["horse_id"])
    assert ped["ancestor_count"] == 62

    assert len(st.load("race_lists", "20261004")["races"]) == 8
    assert st.load("race_day_schedule", "20261010")["slots"]


def test_past_day_has_results_and_upcoming_does_not(mock_storage):
    st = mock_storage
    results = st.list_keys("race_result")
    assert len(results) == 8
    assert all("20261004" not in k and k.startswith("2026") for k in results)
    r = st.load("race_result", results[0])
    assert [e["finish_position"] for e in r["entries"]] == list(range(1, 9))
    assert r["payoff"]["単勝"]["payout"]


def test_save_in_dev_writes_locally_and_survives_clean(mock_storage):
    st = mock_storage
    data = {"race_id": "209901010101", "entries": []}
    assert st.save("race_odds", "209901010101", data) is True
    assert st.load("race_odds", "209901010101") is not None
    make_dev_mock.clean(make_dev_mock.dev_mock_root())
    assert st.list_keys("race_odds") == ["209901010101"]
    assert st.list_keys("race_shutuba") == []


def test_generate_does_not_overwrite_real_page_reference(dev_env):
    real = dev_env / "page_reference" / "race_lists" / "20261004.json"
    real.parent.mkdir(parents=True)
    real.write_text('{"date": "20261004", "races": []}', encoding="utf-8")
    make_dev_mock.generate(TODAY, make_dev_mock.dev_mock_root())
    assert real.read_text(encoding="utf-8") == '{"date": "20261004", "races": []}'


def test_non_dev_env_has_no_dev_store(monkeypatch, tmp_path):
    monkeypatch.setenv("KEIBA_ENV", "stg")
    st = HybridStorage(base_dir=str(tmp_path), bucket_name="")
    assert st._dev_store is None


def test_mock_has_samples_for_every_schema_category(mock_storage, dev_env):
    """スキーマ定義済みの全カテゴリにモックがある（新スキーマを足したらモック生成も足す）。"""
    from src.data_health.checks import check_dev_samples, dev_sample_categories
    from src.scraper import schemas

    st = mock_storage
    for cat in schemas.SCHEMAS:
        keys = st.list_keys(cat)
        assert keys, f"{cat} のモックが無い"
        for k in keys[:3]:
            assert schemas.validate(cat, st.load(cat, k))["passed"], f"{cat}/{k} がスキーマ不適合"
    assert set(schemas.SCHEMAS) <= set(dev_sample_categories())
    assert check_dev_samples(make_dev_mock.dev_mock_root(), dev_env / "page_reference")["status"] == "ok"


def test_derived_categories_match_production_extractors(mock_storage):
    from src.scraper.row_data_extractor import DERIVED_CATEGORY_MAP

    st = mock_storage
    for derived, (parent, fn) in DERIVED_CATEGORY_MAP.items():
        key = st.list_keys(parent)[0]
        src = st.load(parent, key)
        assert {k: v for k, v in st.load(derived, key).items() if k != "_meta"} == fn(src)


def test_tests_dir_schema_only_categories_have_no_hidden_requirement(dev_env):
    """不足を検知できること: カテゴリを消すと dev.mock_samples が FAIL になる。"""
    import shutil

    from src.data_health.checks import check_dev_samples

    make_dev_mock.generate(TODAY, make_dev_mock.dev_mock_root())
    shutil.rmtree(make_dev_mock.dev_mock_root() / "race_paddock")
    r = check_dev_samples(make_dev_mock.dev_mock_root(), dev_env / "page_reference")
    assert r["status"] == "fail" and "race_paddock" in r["detail"]


def test_dev_api_never_takes_scrape_queue_leader(dev_env, monkeypatch):
    """dev の API は運用の取得キュー（data/queue は環境で分かれていない）の処理担当にならない。"""
    from src.api import app as api_app

    opened = []
    monkeypatch.setattr("builtins.open", lambda *a, **k: opened.append(a) or (_ for _ in ()).throw(AssertionError("lock file opened")))
    assert api_app._try_acquire_queue_runner_leader() is False
    assert opened == []


def test_watchdog_keeps_ops_queue_worker_whatever_the_profile():
    # この PC の画面・API を dev にしても、運用の取得キューの処理役は .env の KEIBA_ENV のまま 127.0.0.1 で常駐させる。
    # 止めるときはプロセス番号のファイルだけを使う（名前で探すと dev の API まで止める）
    from pathlib import Path

    s = Path("scripts/server/server_watchdog.sh").read_text(encoding="utf-8")
    body = s[s.index("check_ops_queue() {"):s.index("# MLflow Tracking サーバー")]
    assert "check_ops_queue || true" in s
    assert "kill_by_pattern" not in body and "kill_by_pid_file \"$OPS_QUEUE_PID_FILE\"" in body
    start = s[s.index("start_ops_queue() {"):s.index("check_ops_queue() {")]
    assert 'KEIBA_ENV="$OPS_QUEUE_ENV"' in start and "--host 127.0.0.1" in start
    assert "grep -E '^KEIBA_ENV=' \"${PROJECT_DIR}/.env\"" in s          # 画面・API のプロファイル（dev は KEIBA_ENV=dev を export）に引きずられない
