"""Cloud Run のエントリ（FastAPI + Flask + /ops の振り分け・共有ヘッダ・重い import を避けること）。外部接続なし。"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_route_for(monkeypatch):
    monkeypatch.setenv("KEIBA_CLOUDRUN_LAZY", "1")
    from src.api.cloudrun_entry import Gateway

    r = Gateway.route_for
    assert r("/ops/heartbeat") == "ops" and r("/ops/heartbeat/status") == "ops"
    assert r("/api/v1/health") == "flask" and r("/api/v1") == "flask" and r("/api/v1/races") == "flask"
    assert r("/api/health") == "fastapi" and r("/api/race/1/predictions") == "fastapi" and r("/api/v10") == "fastapi"
    assert r("/data/image/x.png") == "fastapi" and r("/opsx") == "fastapi"


@pytest.fixture
def entry(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "dev")
    monkeypatch.setenv("KEIBA_CLOUDRUN_LAZY", "1")
    monkeypatch.delenv("KEIBA_SERVING_STATE_PULL", raising=False)
    from src.api import cloudrun_entry

    return cloudrun_entry


def test_gateway_serves_fastapi_flask_and_ops(entry):
    from starlette.testclient import TestClient

    c = TestClient(entry.build_app())
    assert c.get("/api/health").status_code == 200
    assert c.get("/api/v1/health").json()["status"] == "ok"
    assert c.get("/ops/heartbeat/status").json()["status"] in ("never", "ok", "stale")


def test_proxy_key_protects_everything_except_health_and_heartbeat_post(entry, monkeypatch):
    from starlette.testclient import TestClient

    monkeypatch.setenv("KEIBA_PROXY_KEY_REQUIRED", "1")
    monkeypatch.setenv("KEIBA_PROXY_KEY", "k-test")
    c = TestClient(entry.build_app())
    assert c.get("/api/health").status_code == 200 and c.get("/api/v1/health").status_code == 200
    for path in ("/api/race/202604040201/predictions", "/api/v1/races", "/ops/heartbeat/status"):
        r = c.get(path)
        assert r.status_code == 401 and r.json()["reason"] == "proxy_key", path
    assert c.get("/ops/heartbeat/status", headers={"X-Keiba-Proxy-Key": "k-test"}).status_code == 200
    # ハートビートの受信は共有ヘッダではなく署名で認証される（署名が無ければアプリ側の 401）
    r = c.post("/ops/heartbeat", content=b"{}")
    assert r.status_code == 401 and r.json().get("reason") != "proxy_key"


def test_requires_explicit_env(monkeypatch):
    monkeypatch.setenv("KEIBA_CLOUDRUN_LAZY", "1")
    monkeypatch.setenv("KEIBA_ENV", "")
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("KEIBA_ALLOW_IMPLICIT_PROD", raising=False)
    from src.api import cloudrun_entry

    with pytest.raises(SystemExit):
        cloudrun_entry.build_app()


def test_no_heavy_libraries_imported_at_startup():
    """起動直後に学習系・取得系の重いライブラリを読み込まない（メモリ 2GiB・起動時間のため）。別プロセスで確認する。"""
    code = (
        "import os,sys;os.environ['KEIBA_ENV']='dev';os.environ['KEIBA_CLOUDRUN_LAZY']='1';"
        "from src.api import cloudrun_entry as c;c.build_app();"
        "bad=[m for m in ('torch','mlflow','optuna','playwright','matplotlib','lightgbm','xgboost','catboost','sklearn','pandas','fitz') if m in sys.modules];"
        "print('BAD:'+','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120).stdout
    assert "BAD:" in out, out
    assert out.strip().splitlines()[-1] == "BAD:", out


def test_cloudrun_requirements_are_subset_of_requirements():
    def names(p):
        return {re.split(r"[<>=!~\[ ]", ln.strip())[0].lower() for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")}

    full, slim = names(ROOT / "requirements.txt"), names(ROOT / "deploy/docker/requirements-cloudrun.txt")
    assert slim - full <= {"a2wsgi"}                     # a2wsgi だけは requirements.txt に無い（wsgi.py と同じ依存）
    for must in ("fastapi", "flask", "uvicorn", "google-cloud-storage", "pandas", "pyarrow", "lightgbm"):
        assert must in slim
    for banned in ("torch", "mlflow", "optuna", "playwright", "pymupdf", "pytest", "ipykernel", "psycopg", "alembic"):
        assert banned not in slim


def test_dockerfile_basics():
    df = (ROOT / "deploy/docker/Dockerfile.cloudrun").read_text(encoding="utf-8")
    assert "src.api.cloudrun_entry:app" in df and "--workers 1" in df and "USER keiba" in df
    assert "COPY .env" not in df and "KEIBA_ENV=" not in df.split("CMD")[0].split("ENV")[-1]   # 環境名はデプロイ時に明示（イメージに焼かない）
    ign = (ROOT / "deploy/docker/Dockerfile.cloudrun.dockerignore").read_text(encoding="utf-8")
    for must in (".env", "data/", ".git/", "tests/"):
        assert must in ign


# ── リクエスト課金（KEIBA_MIRROR_MODE=request）: 取り込みはリクエストをきっかけに ──
class FakeRefresher:
    def __init__(self, wait=False):
        self.wait, self.started, self.wait_s = wait, 0, 1.0
    def should_wait(self): return self.wait
    def start_if_due(self):
        import threading

        self.started += 1
        t = threading.Thread(target=lambda: None); t.start()
        return t


def test_mirror_mode_defaults_to_thread_and_gate_skips_polling_paths(entry, monkeypatch):
    monkeypatch.delenv("KEIBA_MIRROR_MODE", raising=False)
    assert entry.mirror_mode() == "thread"
    monkeypatch.setenv("KEIBA_MIRROR_MODE", "Request")
    assert entry.mirror_mode() == "request"
    for p in ("/api/health", "/api/v1/health", "/api/maintenance/status", "/ops/heartbeat", "/ops"):
        assert not entry.gate_applies(p), p                      # 画面を開いているだけで60秒ごとに来るもの・運用の受信は、取り込みのきっかけにしない
    for p in ("/api/race/1/predictions", "/api/v1/races", "/api/race-list/20261025"):
        assert entry.gate_applies(p), p


def test_gateway_triggers_refresh_only_for_content_requests(entry):
    from starlette.testclient import TestClient

    gw_app = entry.build_app()
    gateway = gw_app.app                                         # ProxyKeyMiddleware が包んでいる Gateway
    fr = FakeRefresher()
    gateway.refresher = fr
    c = TestClient(gw_app)
    c.get("/api/health"); c.get("/api/maintenance/status"); c.get("/ops/heartbeat/status")
    assert fr.started == 0
    c.get("/api/v1/health")                                      # health は取り込まない
    assert fr.started == 0
    c.get("/api/v1/races")
    assert fr.started == 1


def test_gateway_waits_for_pull_only_after_long_gap_and_never_fails_the_request(entry):
    from starlette.testclient import TestClient

    gw_app = entry.build_app()
    gateway = gw_app.app
    calls = []

    class Waiting(FakeRefresher):
        def start_if_due(self):
            calls.append("start")
            return super().start_if_due()
    gateway.refresher = Waiting(wait=True)
    assert TestClient(gw_app).get("/api/v1/races").status_code in (200, 401, 404, 500, 503)    # 取り込みを待った上で通常どおり処理される
    assert calls == ["start"]

    class Broken:
        def should_wait(self): raise RuntimeError("boom")
    gateway.refresher = Broken()
    assert TestClient(gw_app).get("/api/v1/health").status_code == 200                       # 取り込み側の失敗でリクエストは落ちない
    assert TestClient(gw_app).get("/api/v1/races").status_code != 599


def test_request_mode_waits_for_full_initial_pull_at_startup(entry, monkeypatch):
    import asyncio

    from src.ops import gcs_mirror

    gcs_mirror.initial_full_pulled.clear()
    monkeypatch.setattr(entry, "MIRROR_FULL_WAIT_SEC", 0.05)
    asyncio.run(entry._startup(None, warm=False, wait_full=True))          # 時間切れでも例外を出さず起動を続ける
    gcs_mirror.initial_full_pulled.set()
    asyncio.run(entry._startup(None, warm=False, wait_full=True))
