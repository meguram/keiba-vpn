"""GCS の障害の間の HybridStorage（状態の共有・すぐ返す・クライアントの作り直し）。本物の GCS には触れない。
2026-11-15 の 504（docs/operations/api_memory_20261117.md）の再発防止。"""

import threading

import pytest

from src.scraper import storage as S


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    def download_as_text(self, timeout=None, retry=None):
        self.bucket.calls += 1
        if self.bucket.error is not None:
            raise self.bucket.error
        return '{"ok": 1}'


class FakeBucket:
    def __init__(self, error=None):
        self.error, self.calls = error, 0

    def blob(self, name):
        return FakeBlob(self, name)


class SSLEOF(Exception):
    pass


def _err():
    return SSLEOF("HTTPSConnectionPool: Max retries exceeded (Caused by SSLError(UNEXPECTED_EOF_WHILE_READING))")


@pytest.fixture
def make(monkeypatch, tmp_path):
    monkeypatch.setenv("KEIBA_ENV", "prod")
    monkeypatch.setenv("GCS_BUCKET", "bkt-test")
    S._SHARED_GCS.clear()
    S._DEGRADED.update(last_ns=0, count=0)

    def _make(bucket, *, shared="1", fail_fast="1"):
        monkeypatch.setenv("KEIBA_GCS_SHARED_STATE", shared)
        monkeypatch.setenv("KEIBA_GCS_FAIL_FAST", fail_fast)
        st = S.HybridStorage(base_dir=str(tmp_path))
        st._gcs_available, st._gcs_bucket, st._gcs_retry = True, bucket, None
        monkeypatch.setattr(st, "_is_locally_cached", lambda c: False)
        return st

    yield _make
    S._SHARED_GCS.clear()


def test_failure_state_is_shared_between_instances_when_enabled(make):
    b = FakeBucket(_err())
    a = make(b)
    assert a.load("race_result", "202608041104") is None and b.calls == 1
    c = make(b)                                           # 要求ごとに作る新しいインスタンスも、不調を知っている
    assert not c._gcs_healthy
    assert c.load("race_result", "202608041105") is None
    assert b.calls == 1                                   # GCS を呼ばずにすぐ返した


def test_without_shared_state_each_instance_starts_healthy(make):
    b = FakeBucket(_err())
    a = make(b, shared="0")
    a.load("race_result", "1")
    c = make(FakeBucket(), shared="0")
    assert c._gcs_healthy and c.load("race_result", "2") == {"ok": 1}


def test_fail_fast_off_keeps_old_behavior(make):
    b = FakeBucket(_err())
    a = make(b, fail_fast="0")
    a.load("race_result", "1")
    a.load("race_result", "2")
    assert b.calls == 2                                   # ホストの cron・Jobs は従来どおり読みに行く


def test_fail_fast_result_is_not_cached_and_probe_recovers(make, monkeypatch):
    b = FakeBucket(_err())
    st = make(b)
    st.load("race_result", "1")
    assert st.load("race_result", "1") is None
    assert "race_result/1" not in st._load_cache          # 障害の None はキャッシュに入れない
    b.error = None
    st._gcs.last_probe = 0.0                              # 試しの読み出しの時刻が来た（最後の失敗から 15 秒以上）
    st._gcs.last_failure -= 20
    assert st.load("race_result", "1") == {"ok": 1}
    assert st._gcs_healthy                                # 1 件成功したら戻る
    assert st.load("race_result", "3") == {"ok": 1}


def test_fail_fast_marks_request_and_degraded_time(make):
    st = make(FakeBucket(_err()))
    tok = S.begin_request_flag()
    try:
        before = S._DEGRADED["count"]
        st.load("race_result", "1")
        st.load("race_result", "2")
        assert S.request_flag().get("failfast") is True
        assert S._DEGRADED["count"] > before and S.degraded_since(0)
    finally:
        S.end_request_flag(tok)
    assert S.request_flag() is None


def test_request_flag_reaches_worker_threads(make):
    import asyncio

    st = make(FakeBucket(_err()))

    async def main():
        tok = S.begin_request_flag()
        try:
            await asyncio.to_thread(st.load, "race_result", "1")
            return dict(S.request_flag())
        finally:
            S.end_request_flag(tok)

    assert asyncio.run(main()).get("failfast") is True


def test_client_is_rebuilt_after_repeated_transport_errors(make, monkeypatch):
    monkeypatch.setenv("KEIBA_GCS_REBUILD_AFTER", "3")
    st = make(FakeBucket(_err()))
    for _ in range(2):
        st._mark_gcs_failure(_err())
    assert st._gcs_bucket is not None
    st._mark_gcs_failure(_err())
    assert st._gcs_bucket is None and st._gcs.rebuilds == 1         # 次の読み出しでクライアントから作り直す
    st._mark_gcs_failure(ValueError("bad json"))                     # 通信以外の失敗は数えない
    assert st._gcs.transport_errors == []


def test_not_found_is_not_a_failure(make):
    class NotFound(Exception):
        pass

    b = FakeBucket(NotFound("404 No such object"))
    st = make(b)
    import src.scraper.storage as mod

    orig = mod._is_gcs_not_found
    mod._is_gcs_not_found = lambda e: isinstance(e, NotFound)
    try:
        assert st.load("race_result", "x") is None
        assert st._gcs_healthy
    finally:
        mod._is_gcs_not_found = orig


def test_timeouts_follow_env(make, monkeypatch):
    monkeypatch.setenv("KEIBA_GCS_TIMEOUT_SEC", "4")
    st = make(FakeBucket())
    assert st._GCS_TIMEOUT == 4.0
    monkeypatch.setenv("KEIBA_GCS_RETRY_DEADLINE_SEC", "3")
    r = S.HybridStorage._build_retry()
    assert getattr(r, "_deadline", getattr(r, "_timeout", None)) == 3.0


def test_shared_state_is_thread_safe(make):
    b = FakeBucket(_err())
    insts = [make(b) for _ in range(4)]
    ts = [threading.Thread(target=lambda s=s: [s._mark_gcs_failure(_err()) for _ in range(20)]) for s in insts]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len({id(s._gcs) for s in insts}) == 1


def test_non_transport_errors_do_not_stop_the_shared_state(make):
    """壊れた JSON・403 など 1 件の失敗では、共有の状態ごと止めない（すぐ返す動きのとき）。"""
    b = FakeBucket(PermissionError("403 Forbidden"))
    st = make(b)
    assert st.load("race_result", "1") is None
    assert st._gcs_healthy and not S.gcs_unhealthy_now()
    b.error = None
    assert make(b).load("race_result", "2") == {"ok": 1}


def test_host_mode_keeps_old_backoff_and_no_rebuild(make, monkeypatch):
    monkeypatch.setenv("KEIBA_GCS_REBUILD_AFTER", "1")
    st = make(FakeBucket(), shared="0", fail_fast="0")
    st._mark_gcs_failure(ValueError("x"))
    assert not st._gcs_healthy                            # ホストの cron・Jobs は従来どおり、どの失敗でも 60 秒の退避
    st._mark_gcs_failure(_err())
    assert st._gcs.rebuilds == 0 and st._gcs_bucket is not None


def test_gcs_unhealthy_now_reads_shared_state_only(make):
    st = make(FakeBucket(_err()))
    assert not S.gcs_unhealthy_now()
    st.load("race_result", "1")
    assert S.gcs_unhealthy_now()
    S._SHARED_GCS.clear()
    assert not S.gcs_unhealthy_now()


def test_stale_copy_does_not_mark_request(make, monkeypatch):
    st = make(FakeBucket(_err()))
    monkeypatch.setattr(st, "_is_locally_cached", lambda c: True)
    monkeypatch.setattr(st, "_read_local_cache", lambda c, k, allow_stale=False: {"stale": 1} if allow_stale else None)
    monkeypatch.setattr(st, "_weekly_disk_l2_maybe_write", lambda *a: None)
    tok = S.begin_request_flag()
    try:
        assert st.load("race_result", "1") == {"stale": 1}
        assert not S.request_flag().get("failfast")       # 写しで返せた要求は 503 にしない
    finally:
        S.end_request_flag(tok)


def test_get_bucket_never_returns_none_during_rebuild(make):
    st = make(FakeBucket())
    b = st._get_bucket()
    st._gcs.bucket = None                                 # 別のスレッドが作り直しで消した直後
    st._gcs.client = type("C", (), {"bucket": lambda self, name: b})()
    import src.scraper.storage as mod

    orig = mod.HybridStorage._build_credentials
    mod.HybridStorage._build_credentials = staticmethod(lambda: None)
    try:
        import google.cloud.storage as gcs_lib

        orig_client = gcs_lib.Client
        gcs_lib.Client = lambda *a, **k: type("C", (), {"bucket": lambda self, name: b})()
        try:
            assert st._get_bucket() is b
        finally:
            gcs_lib.Client = orig_client
    finally:
        mod.HybridStorage._build_credentials = orig
