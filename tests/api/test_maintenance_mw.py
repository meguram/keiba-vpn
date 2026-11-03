"""メンテナンス時間帯のゲート（src/api/maintenance_mw.py）。FastAPI のミドルウェアと Flask の before_request。"""

import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from flask import Flask, jsonify

from src.api import maintenance_mw as MW
from src.utils import maintenance as M

ENV_ON = {"KEIBA_ENV": "dev", "KEIBA_MAINTENANCE_WINDOW": "1", "KEIBA_MAINTENANCE_DEV": "1",
          "KEIBA_MAINTENANCE_BYPASS_TOKEN": "tok-123"}


def make_fastapi(admin=False):
    app = FastAPI()
    MW.install_fastapi(app, lambda request: admin)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/race/{rid}/predictions")
    def pred(rid: str):
        return {"race_id": rid}

    @app.get("/api/scrape-jobs")
    def jobs():
        return {"jobs": []}

    @app.get("/weekly")
    def page():
        return {"page": True}

    return app


def run_at(now_str, env=None, **kw):
    e = dict(ENV_ON, KEIBA_MAINTENANCE_NOW=now_str)
    e.update(env or {})
    p = mock.patch.dict("os.environ", e, clear=False)
    return p


class TestFastApi(unittest.TestCase):
    def setUp(self):
        M.clear_cache()
        MW._audit_seen.clear()
        M._last_phase["phase"] = None

    def get(self, path, now, admin=False, headers=None, env=None, client_host=None):
        with run_at(now, env):
            c = TestClient(make_fastapi(admin), client=(client_host, 50000) if client_host else ("203.0.113.9", 50000))
            return c.get(path, headers=headers or {})

    def test_503_during_window(self):
        r = self.get("/api/race/202604040201/predictions", "2026-10-21 03:10")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["maintenance"], True)
        self.assertEqual(r.json()["ends_at_label"], "04:00")
        self.assertEqual(r.headers["retry-after"], str(50 * 60))
        self.assertIn("no-store", r.headers["cache-control"])
        self.assertEqual(r.headers["x-keiba-maintenance"], "1")

    def test_html_page_gets_html_503(self):
        r = self.get("/weekly", "2026-10-21 03:10", headers={"accept": "text/html"})
        self.assertEqual(r.status_code, 503)
        self.assertIn("メンテナンス中", r.text)
        self.assertIn("noindex", r.text)

    def test_outside_window_passes(self):
        for now in ("2026-10-21 02:59:59", "2026-10-21 04:00:00", "2026-10-21 12:00"):
            self.assertEqual(self.get("/api/race/x/predictions", now).status_code, 200, now)
        self.assertEqual(self.get("/api/race/x/predictions", "2026-10-21 03:00:00").status_code, 503)
        self.assertEqual(self.get("/api/race/x/predictions", "2026-10-21 03:59:59").status_code, 503)

    def test_exempt_paths_pass(self):
        self.assertEqual(self.get("/api/health", "2026-10-21 03:10").status_code, 200)
        self.assertEqual(self.get("/api/scrape-jobs", "2026-10-21 03:10").status_code, 200)

    def test_status_endpoint_passes_and_reports(self):
        r = self.get("/api/maintenance/status", "2026-10-21 03:10")
        self.assertEqual(r.status_code, 200)
        j = r.json()
        self.assertTrue(j["active"])
        self.assertEqual(j["phase"], "active")
        self.assertEqual(j["remaining_sec"], 50 * 60)
        self.assertIn("no-store", r.headers["cache-control"])
        n = self.get("/api/maintenance/status", "2026-10-21 02:45").json()
        self.assertEqual(n["phase"], "notice")
        self.assertEqual(n["starts_in_sec"], 15 * 60)

    def test_bypass_admin_token_internal(self):
        self.assertEqual(self.get("/api/race/x/predictions", "2026-10-21 03:10", admin=True).status_code, 200)
        self.assertEqual(self.get("/api/race/x/predictions", "2026-10-21 03:10",
                                  headers={"X-Keiba-Maintenance-Bypass": "tok-123"}).status_code, 200)
        self.assertEqual(self.get("/api/race/x/predictions", "2026-10-21 03:10",
                                  headers={"X-Keiba-Maintenance-Bypass": "nope"}).status_code, 503)
        # 同一ホストの直接呼び出し（cron・ランナー）は通る。プロキシ経由（XFF 付き）は止まる
        self.assertEqual(self.get("/api/race/x/predictions", "2026-10-21 03:10", client_host="127.0.0.1").status_code, 200)
        self.assertEqual(self.get("/api/race/x/predictions", "2026-10-21 03:10", client_host="127.0.0.1",
                                  headers={"X-Forwarded-For": "203.0.113.9"}).status_code, 503)

    def test_flag_off_is_passthrough(self):
        with mock.patch.dict("os.environ", {"KEIBA_ENV": "stg", "KEIBA_MAINTENANCE_NOW": "2026-10-21 03:10"}, clear=False):
            import os
            os.environ.pop("KEIBA_MAINTENANCE_WINDOW", None)
            c = TestClient(make_fastapi())
            self.assertEqual(c.get("/api/race/x/predictions").status_code, 200)

    def test_clock_override_ignored_outside_dev(self):
        # stg では ?at= も環境変数の時計も無視する（実時刻で判定）
        with mock.patch.dict("os.environ", {"KEIBA_ENV": "stg", "KEIBA_MAINTENANCE_WINDOW": "1", "KEIBA_MAINTENANCE_NOW": "2026-10-21 03:10"}, clear=False):
            c = TestClient(make_fastapi())
            j = c.get("/api/maintenance/status", params={"at": "2026-10-21 03:10"}).json()
            self.assertNotEqual(j["now"][:16], "2026-10-21T03:10")

    def test_judgement_error_falls_back_to_normal(self):
        with run_at("2026-10-21 03:10"), mock.patch.object(MW, "decide", side_effect=RuntimeError("boom")):
            c = TestClient(make_fastapi())
            self.assertEqual(c.get("/api/race/x/predictions").status_code, 200)

    def test_dev_bypass_can_be_disabled(self):
        # 管理者判定が True でも KEIBA_MAINTENANCE_DEV_BYPASS=0 のときは Cookie の署名検証だけを見る（Cookie 無しなら止まる）
        r = self.get("/api/race/x/predictions", "2026-10-21 03:10", admin=True, env={"KEIBA_MAINTENANCE_DEV_BYPASS": "0"})
        self.assertEqual(r.status_code, 503)


class TestFlask(unittest.TestCase):
    def setUp(self):
        M.clear_cache()
        MW._audit_seen.clear()

    def app(self, admin=False):
        app = Flask(__name__)
        MW.install_flask(app, lambda: admin)

        @app.get("/api/v1/races")
        def races():
            return jsonify({"races": []})

        @app.get("/api/v1/health")
        def health():
            return jsonify({"status": "ok"})

        return app

    def test_flask_gate(self):
        with run_at("2026-10-21 03:10"):
            c = self.app().test_client()
            r = c.get("/api/v1/races", environ_overrides={"REMOTE_ADDR": "203.0.113.9"})
            self.assertEqual(r.status_code, 503)
            self.assertEqual(r.get_json()["maintenance"], True)
            self.assertEqual(r.headers["Retry-After"], str(50 * 60))
            self.assertEqual(c.get("/api/v1/health", environ_overrides={"REMOTE_ADDR": "203.0.113.9"}).status_code, 200)
            self.assertEqual(c.get("/api/v1/maintenance/status", environ_overrides={"REMOTE_ADDR": "203.0.113.9"}).get_json()["active"], True)
            self.assertEqual(self.app(admin=True).test_client().get("/api/v1/races", environ_overrides={"REMOTE_ADDR": "203.0.113.9"}).status_code, 200)

    def test_flask_outside_window(self):
        with run_at("2026-10-21 12:00"):
            r = self.app().test_client().get("/api/v1/races", environ_overrides={"REMOTE_ADDR": "203.0.113.9"})
            self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()

