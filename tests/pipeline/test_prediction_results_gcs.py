import json

import pytest

from src.pipeline.inference import prediction_results as pr


class FakeGcs:
    def __init__(self):
        self.objs: dict[str, str] = {}
        self.gets = 0

    def install(self, monkeypatch):
        monkeypatch.setattr(pr, "_gcs_get", self.get)
        monkeypatch.setattr(pr, "_gcs_put", self.put)
        monkeypatch.setattr(pr, "_gcs_list", lambda: sorted(self.objs))

    def get(self, d):
        self.gets += 1
        return self.objs.get(d)

    def put(self, d, text):
        self.objs[d] = text


def _doc(date, status="final"):
    return {"date": date, "status": status, "summary": pr.summarize([]), "highlights": [], "races": []}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "store_dir", lambda: tmp_path)
    monkeypatch.setenv("GCS_BUCKET", "b")
    monkeypatch.setenv("KEIBA_PREDICTION_RESULTS_GCS", "1")
    monkeypatch.setattr("src.config.gcp_guard.gcp_forbidden", lambda: False)
    pr.clear_remote_cache()
    g = FakeGcs()
    g.install(monkeypatch)
    return g


def test_disabled_by_default_and_in_dev(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "store_dir", lambda: tmp_path)
    monkeypatch.setenv("GCS_BUCKET", "b")
    monkeypatch.delenv("KEIBA_PREDICTION_RESULTS_GCS", raising=False)
    monkeypatch.delenv("KEIBA_RUN_LOCATION", raising=False)
    monkeypatch.setattr("src.config.gcp_guard.gcp_forbidden", lambda: False)
    assert not pr.gcs_enabled()
    monkeypatch.setenv("KEIBA_RUN_LOCATION", "gcp")
    assert pr.gcs_enabled()                                   # Cloud Run / Jobs では自動で有効
    monkeypatch.setenv("KEIBA_PREDICTION_RESULTS_GCS", "0")
    assert not pr.gcs_enabled()                               # 明示的な無効
    monkeypatch.setenv("KEIBA_PREDICTION_RESULTS_GCS", "1")
    monkeypatch.setattr("src.config.gcp_guard.gcp_forbidden", lambda: True)
    assert not pr.gcs_enabled()                               # dev は GCP 禁止


def test_write_day_also_uploads_and_a_fresh_instance_reads_it_back(env, tmp_path):
    pr.write_day(_doc("20261024"))
    assert json.loads(env.objs["20261024"])["date"] == "20261024"
    for f in tmp_path.glob("*.json"):                          # Cloud Run 相当: ローカルが空
        f.unlink()
    pr.clear_remote_cache()
    assert pr.read_day("20261024")["status"] == "final"
    assert pr.list_days() == ["20261024"]


def test_final_docs_are_cached_and_gcs_failure_never_breaks_local_write(env, tmp_path, monkeypatch):
    env.objs["20261017"] = json.dumps(_doc("20261017"))
    pr.read_day("20261017")
    pr.read_day("20261017")
    assert env.gets == 1                                       # 確定済みは再取得しない
    def boom(d, text):
        raise RuntimeError("gcs down")
    monkeypatch.setattr(pr, "_gcs_put", boom)
    assert pr.write_day(_doc("20261025")) is not None          # ローカルには残る
    assert (tmp_path / "20261025.json").exists()


def test_local_wins_and_explicit_base_never_touches_gcs(env, tmp_path):
    env.objs["20261024"] = json.dumps({**_doc("20261024"), "status": "provisional"})
    (tmp_path / "20261024.json").write_text(json.dumps(_doc("20261024")), encoding="utf-8")
    assert pr.read_day("20261024")["status"] == "final"       # ローカルが先
    other = tmp_path / "x"
    other.mkdir()
    assert pr.read_day("20261024", other) is None and env.gets == 0
    assert pr.list_days(other) == []


def test_sync_to_gcs_uploads_every_local_day(env, tmp_path):
    for d in ("20261010", "20261011"):
        (tmp_path / f"{d}.json").write_text(json.dumps(_doc(d)), encoding="utf-8")
    assert pr.sync_to_gcs() == {"uploaded": 2} and sorted(env.objs) == ["20261010", "20261011"]


def test_read_days_reads_remote_days_in_order_and_skips_missing(env):
    for d in ("20261010", "20261012", "20261011"):
        env.objs[d] = json.dumps(_doc(d))
    out = pr.read_days(["20261010", "20261011", "20261099", "20261012"])
    assert [x["date"] for x in out] == ["20261010", "20261011", "20261012"]
