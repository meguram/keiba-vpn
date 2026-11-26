"""案 A: 起動処理（取り込み・ウォームアップ）を裏で進め、ログインまわり・ヘルスは先に答える。外部接続なし。"""

import asyncio

import pytest


@pytest.fixture
def entry(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "dev")
    monkeypatch.setenv("KEIBA_CLOUDRUN_LAZY", "1")
    monkeypatch.delenv("KEIBA_SERVING_STATE_PULL", raising=False)
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    monkeypatch.delenv("KEIBA_EARLY_AUTH", raising=False)
    from src.api import auth
    from src.api import cloudrun_entry

    monkeypatch.setattr(auth, "members_refresh_hook", None)
    return cloudrun_entry


async def _start(app):
    """lifespan を動かし、startup.complete が返るまで待つ。戻り値は (lifespan のタスク, 受信の箱)。"""
    q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()

    async def send(msg):
        if msg["type"] == "lifespan.startup.complete":
            done.set()

    await q.put({"type": "lifespan.startup"})
    task = asyncio.create_task(app({"type": "lifespan"}, q.get, send))
    await asyncio.wait_for(done.wait(), 5)
    return task, q


async def _get(app, path, timeout=5.0):
    import httpx

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        return await asyncio.wait_for(c.get(path), timeout)


def test_port_opens_before_startup_and_auth_answers_first(entry):
    async def run():
        gw_app = entry.build_app()
        gw = gw_app.app
        release = asyncio.Event()
        order = []

        async def slow_startup():
            await release.wait()
            order.append("startup-done")

        gw.warmup = slow_startup
        task, q = await _start(gw_app)                       # 起動処理が済む前に startup.complete（= ポートが開く）
        assert gw.ready is not None and not gw.ready.is_set()
        r = await _get(gw_app, "/api/v1/auth/status")        # ログイン状態は起動処理を待たない
        assert r.status_code == 200 and 'inst;desc="age=' in r.headers.get("server-timing", "")
        assert (await _get(gw_app, "/api/v1/health")).status_code == 200

        content = asyncio.create_task(_get(gw_app, "/api/v1/races", timeout=10))
        await asyncio.sleep(0.2)
        assert not content.done()                            # 内容の要求は起動処理の完了を待つ
        release.set()
        r = await content
        assert order == ["startup-done"] and r.status_code != 599
        await q.put({"type": "lifespan.shutdown"})
        await task

    asyncio.run(run())


def test_warmup_requests_to_itself_do_not_wait_for_themselves(entry, monkeypatch):
    async def run():
        gw_app = entry.build_app()
        gw = gw_app.app
        seen = []

        async def warm_calls_itself():
            entry._IN_WARMUP.set(True)                       # _warmup と同じ印
            r = await _get(gw_app, "/api/v1/races", timeout=5)
            seen.append(r.status_code)

        gw.warmup = warm_calls_itself
        task, q = await _start(gw_app)
        await asyncio.wait_for(gw.ready.wait(), 6)           # 自分自身を待って止まらない
        assert seen and seen[0] != 599
        await q.put({"type": "lifespan.shutdown"})
        await task

    asyncio.run(run())


def test_content_wait_is_capped(entry, monkeypatch):
    monkeypatch.setattr(entry, "READY_WAIT_SEC", 0.2)

    async def run():
        gw_app = entry.build_app()
        gw = gw_app.app

        async def never():
            await asyncio.Event().wait()

        gw.warmup = never
        task, q = await _start(gw_app)
        r = await _get(gw_app, "/api/v1/races", timeout=3)   # 上限まで待っても済まなければ、温まっていない応答を返さず 503
        assert r.status_code == 503 and r.headers["cache-control"] == "no-store" and r.headers["retry-after"]
        assert (await _get(gw_app, "/api/v1/maintenance/status")).status_code != 503      # メンテの状態・/ops は待たない
        gw._startup_task.cancel()
        await q.put({"type": "lifespan.shutdown"})
        await task

    asyncio.run(run())


def test_early_auth_off_keeps_old_behavior(entry, monkeypatch):
    monkeypatch.setenv("KEIBA_EARLY_AUTH", "0")

    async def run():
        gw_app = entry.build_app()
        gw = gw_app.app
        order = []

        async def startup():
            await asyncio.sleep(0.1)
            order.append("startup-done")

        gw.warmup = startup
        task, q = await _start(gw_app)                       # 従来: 起動処理が済んでから startup.complete
        assert order == ["startup-done"] and gw.ready is None
        await q.put({"type": "lifespan.shutdown"})
        await task

    asyncio.run(run())


def test_initial_pull_takes_members_first_and_sets_the_flag(tmp_path, monkeypatch):
    from src.ops import gcs_mirror

    calls = []
    monkeypatch.setattr(gcs_mirror, "refresh_members", lambda root, be, **k: calls.append(("members", k.get("min_interval"))) or True)
    monkeypatch.setattr(gcs_mirror, "_initial_from_bundle", lambda be, root, d: calls.append("bundle") or {"bundle_files": 0})
    gcs_mirror.initial_members_pulled.clear()
    gcs_mirror._initial_pull(object(), tmp_path, 2.0)
    assert calls == [("members", 0.0), "bundle"] and gcs_mirror.initial_members_pulled.is_set()    # 起動時は間隔の制限を受けない

    gcs_mirror.initial_members_pulled.clear()

    def boom(root, be, **k):
        raise RuntimeError("gcs down")

    monkeypatch.setattr(gcs_mirror, "refresh_members", boom)
    gcs_mirror._initial_pull(object(), tmp_path, 2.0)        # 名簿の取り込みに失敗しても、印は立てて先へ進む
    assert gcs_mirror.initial_members_pulled.is_set()


def test_shutdown_during_startup_cancels_the_background_task(entry):
    async def run():
        gw_app = entry.build_app()
        gw = gw_app.app

        async def never():
            await asyncio.Event().wait()

        gw.warmup = never
        task, q = await _start(gw_app)
        await q.put({"type": "lifespan.shutdown"})
        await asyncio.wait_for(task, 3)                      # 起動処理の途中でも、shutdown は止まらずに終わる
        assert gw._startup_task.cancelled()

    asyncio.run(run())


def test_startup_refresh_failure_does_not_block_the_next_login_refresh(tmp_path, monkeypatch):
    from src.ops import gcs_mirror

    class Down:
        def download(self, key):
            raise RuntimeError("gcs down")

    class Up:
        def download(self, key):
            return b'{"users": []}'

    monkeypatch.setitem(gcs_mirror._members_refresh, "at", -1e18)
    gcs_mirror._MISS.clear()
    assert not gcs_mirror.refresh_members(tmp_path, Down(), min_interval=0.0)          # 起動時に取れなかった
    assert gcs_mirror.refresh_members(tmp_path, Up())                                   # 直後のログインの取り直しは、60 秒の制限で止まらない


def test_ready_path_reports_startup_completion_without_proxy_key(entry, monkeypatch):
    """起動プローブの道 /ops/ready: 起動処理が済むまで 503、済めば 200。共有鍵が必須でも鍵なしで答える（Cloud Run のプローブは鍵を持たない）。"""
    monkeypatch.setenv("KEIBA_PROXY_KEY_REQUIRED", "1")
    monkeypatch.setenv("KEIBA_PROXY_KEY", "k-test")

    async def run():
        gw_app = entry.build_app()
        gw = gw_app.app
        release = asyncio.Event()

        async def slow_startup():
            await release.wait()

        gw.warmup = slow_startup
        task, q = await _start(gw_app)
        r = await _get(gw_app, entry.READY_PATH, timeout=2)           # 起動処理を待たずにすぐ答える
        assert r.status_code == 503 and r.json() == {"ready": False} and r.headers["cache-control"] == "no-store"
        assert (await _get(gw_app, "/ops/memory", timeout=2)).status_code == 401       # ほかの /ops は今どおり鍵が要る
        release.set()
        await asyncio.wait_for(gw.ready.wait(), 5)
        r = await _get(gw_app, entry.READY_PATH, timeout=2)
        assert r.status_code == 200 and r.json() == {"ready": True}
        await q.put({"type": "lifespan.shutdown"})
        await task

    asyncio.run(run())


def test_deploy_script_uses_http_startup_probe_on_ready_path():
    from pathlib import Path

    from src.api import cloudrun_entry

    s = (Path(__file__).resolve().parents[2] / "scripts" / "gcp" / "deploy_cloud_run_api.sh").read_text(encoding="utf-8")
    assert f"httpGet.path={cloudrun_entry.READY_PATH}" in s and '--startup-probe "$STARTUP_PROBE"' in s
    assert "periodSeconds=4" in s and "failureThreshold=60" in s                  # 4 × 60 = 240 秒（Cloud Run の上限）に収める


def test_preload_models_runs_each_step_and_survives_failures(entry, monkeypatch):
    """起動処理の先読み: 追走の v2・ペースの型・めぐ指数の年の表を読む。1 つが失敗しても残りを読み、起動は続ける。"""
    from src.pipeline.inference import expected_megu, pace_type, tracking_v2_service

    calls = []
    monkeypatch.setattr(tracking_v2_service, "tv2_load_model", lambda: calls.append("tv2"))
    monkeypatch.setattr(pace_type, "_boosters", lambda: (_ for _ in ()).throw(RuntimeError("no model")))
    monkeypatch.setattr(expected_megu, "runs_for_horse", lambda hid, d: calls.append(("megu", hid, d)) or ([], ""))
    took = entry.preload_models("20261123")
    assert calls == ["tv2", ("megu", "", "20261123")]
    assert {"tracking_v2", "pace_type", "expected_megu", "megu_models", "hem_jockey"} <= set(took)
    calls.clear()
    assert "expected_megu" not in entry.preload_models("20261123", part="models")   # config/ の分は data/ を読まない（束より先に流す）
    assert calls == ["tv2"]
    calls.clear()
    took = entry.preload_models("20261123", part="data")
    assert calls == [("megu", "", "20261123")] and "tracking_v2" not in took


def _bundle_pulled(entry, monkeypatch, ok: bool):
    """束の取り込みの印を、このテストだけの Event に差し替える（プロセス全体の印を汚さない）。
    ok=False は、束が読めず一覧の取り込みも失敗した起動（直近分・全体の印は成否を問わず立つが、束の成功の印は立たない）。"""
    import threading

    from src.ops import gcs_mirror

    recent, whole, bundle = threading.Event(), threading.Event(), threading.Event()
    recent.set()
    whole.set()
    if ok:
        bundle.set()
    monkeypatch.setattr(gcs_mirror, "initial_recent_pulled", recent)
    monkeypatch.setattr(gcs_mirror, "initial_full_pulled", whole)
    monkeypatch.setattr(gcs_mirror, "initial_bundle_ok", bundle)
    monkeypatch.setattr(entry, "mirror_pull_enabled", lambda: True)
    monkeypatch.setattr(entry, "_start_mirror_pull", lambda: None)          # GCS の取り込みのスレッドは立てない


@pytest.mark.parametrize("flag,ok,waits", [("", True, False), ("0", True, True), ("", False, True)])
def test_ready_before_warmup_when_bundle_pulled_everything(entry, monkeypatch, flag, ok, waits):
    """束と追いつきが成功した起動は、ウォームアップを流さずに内容の要求を通す（既定）。
    KEIBA_READY_BEFORE_WARMUP=0・束の道が失敗した起動（全体の印は立つが束の成功の印は立たない）は従来どおり待つ。"""
    monkeypatch.setenv("KEIBA_READY_BEFORE_WARMUP", flag)
    _bundle_pulled(entry, monkeypatch, ok)
    preloaded = []
    monkeypatch.setattr(entry, "preload_models", lambda today=None, part="all": preloaded.append(part) or {})
    from src.api import preload_gate

    async def run():
        gw_app = entry.build_app()
        gw = gw_app.app
        release = asyncio.Event()

        async def slow_startup():
            await release.wait()

        gw.warmup = slow_startup
        task, q = await _start(gw_app)
        assert gw.ready.is_set() is (not waits)
        r = await _get(gw_app, entry.READY_PATH, timeout=2)
        assert r.status_code == (503 if waits else 200)
        content = asyncio.create_task(_get(gw_app, "/api/v1/races", timeout=10))
        await asyncio.sleep(0.2)
        assert content.done() is (not waits)
        release.set()
        await content
        await asyncio.wait_for(gw._startup_task, 5)
        assert preloaded[0] == "models"                             # config/ の分は束と並べて最初に（どちらの道でも）
        assert preloaded[1:] == ([] if waits else ["data"])          # 早く通すときは、ウォームアップの代わりに data/ の先読みだけ
        assert not preload_gate.pending()                            # 予測の待ちの印は必ず下りる
        await q.put({"type": "lifespan.shutdown"})
        await task

    asyncio.run(run())
