"""毎日のメンテナンス時間帯の判定（src/utils/maintenance.py）。時計は注入し、OS のタイムゾーンに依存しない。"""

import json
import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from src.utils import maintenance as M

JST = ZoneInfo("Asia/Tokyo")


def at(h, m=0, s=0, day=21):
    return datetime(2026, 10, day, h, m, s, tzinfo=JST)


CFG = M.DEFAULT_CONFIG


class TestStatus(unittest.TestCase):
    def st(self, now, cfg=CFG, enabled=True):
        return M.compute_status(cfg, now, enabled=enabled)

    def test_boundaries(self):
        self.assertFalse(self.st(at(2, 59, 59))["active"])
        s = self.st(at(3, 0, 0))
        self.assertTrue(s["active"])
        self.assertEqual(s["phase"], "active")
        self.assertEqual(s["remaining_sec"], 3600)
        self.assertEqual(self.st(at(3, 59, 59))["remaining_sec"], 1)
        self.assertTrue(self.st(at(3, 59, 59))["active"])
        self.assertFalse(self.st(at(4, 0, 0))["active"])

    def test_notice_window(self):
        self.assertEqual(self.st(at(2, 29, 59))["phase"], "normal")
        s = self.st(at(2, 30, 0))
        self.assertEqual(s["phase"], "notice")
        self.assertEqual(s["starts_in_sec"], 1800)
        self.assertEqual(s["ends_at_label"], "04:00")
        self.assertTrue(s["message"])
        self.assertEqual(self.st(at(2, 59, 59))["phase"], "notice")

    def test_after_window_points_to_next_day(self):
        s = self.st(at(4, 0, 0))
        self.assertEqual(s["phase"], "normal")
        self.assertTrue(s["window_start"].startswith("2026-10-22T03:00"))
        self.assertEqual(s["starts_in_sec"], 23 * 3600)

    def test_utc_clock_is_converted_to_jst(self):
        # UTC 18:00 = JST 03:00（ホストの OS 時刻は UTC）
        utc = datetime(2026, 10, 20, 18, 0, 0, tzinfo=timezone.utc)
        self.assertTrue(self.st(utc)["active"])
        self.assertFalse(self.st(datetime(2026, 10, 20, 19, 0, 0, tzinfo=timezone.utc))["active"])
        # UTC 03:00 = JST 12:00 は対象外（UTC の 3 時ではない）
        self.assertFalse(self.st(datetime(2026, 10, 21, 3, 0, 0, tzinfo=timezone.utc))["active"])

    def test_naive_now_not_used_as_utc(self):
        # 日跨ぎ: JST 23:30 → 次は翌日 03:00
        s = self.st(at(23, 30))
        self.assertEqual(s["phase"], "normal")
        self.assertEqual(s["starts_in_sec"], int(3.5 * 3600))

    def test_overnight_window_config(self):
        cfg = M._merge(CFG, {"daily": {"start": "23:30", "end": "00:30"}})
        self.assertTrue(M.compute_status(cfg, at(23, 45), enabled=True)["active"])
        self.assertTrue(M.compute_status(cfg, at(0, 10, day=22), enabled=True)["active"])
        self.assertFalse(M.compute_status(cfg, at(0, 30, day=22), enabled=True)["active"])

    def test_disabled_never_active(self):
        s = self.st(at(3, 30), enabled=False)
        self.assertFalse(s["active"])
        self.assertEqual(s["phase"], "normal")

    def test_override(self):
        cfg = M._merge(CFG, {"override": {"start": "2026-10-21 13:00", "end": "2026-10-21 14:30", "message": "臨時メンテです"}})
        s = M.compute_status(cfg, at(13, 10), enabled=True)
        self.assertTrue(s["active"])
        self.assertEqual(s["source"], "override")
        self.assertEqual(s["message"], "臨時メンテです")
        self.assertEqual(s["ends_at_label"], "14:30")
        self.assertEqual(s["retry_after"], 80 * 60)
        # 臨時の開始 30 分前は事前告知
        self.assertEqual(M.compute_status(cfg, at(12, 40), enabled=True)["phase"], "notice")
        # 終わったら通常の窓に戻る
        self.assertFalse(M.compute_status(cfg, at(14, 31), enabled=True)["active"])
        # オフセット付き ISO も読める
        cfg2 = M._merge(CFG, {"override": {"start": "2026-10-21T13:00:00+09:00", "end": "2026-10-21T14:00:00+09:00"}})
        self.assertTrue(M.compute_status(cfg2, at(13, 30), enabled=True)["active"])

    def test_bad_override_ignored(self):
        cfg = M._merge(CFG, {"override": {"start": "garbage", "end": "2026-10-21 14:00"}})
        self.assertFalse(M.compute_status(cfg, at(13, 30), enabled=True)["active"])


class TestEnableAndConfig(unittest.TestCase):
    def test_flag_default_off(self):
        self.assertFalse(M.is_enabled(CFG, {"KEIBA_ENV": "stg"}))
        self.assertTrue(M.is_enabled(CFG, {"KEIBA_ENV": "stg", "KEIBA_MAINTENANCE_WINDOW": "1"}))
        self.assertTrue(M.is_enabled(CFG, {"KEIBA_MAINTENANCE_WINDOW": "1"}))  # KEIBA_ENV 未設定は prod 扱い

    def test_dev_needs_extra_flag(self):
        self.assertFalse(M.is_enabled(CFG, {"KEIBA_ENV": "dev", "KEIBA_MAINTENANCE_WINDOW": "1"}))
        self.assertTrue(M.is_enabled(CFG, {"KEIBA_ENV": "dev", "KEIBA_MAINTENANCE_WINDOW": "1", "KEIBA_MAINTENANCE_DEV": "1"}))

    def test_config_enabled_false(self):
        cfg = M._merge(CFG, {"enabled": False})
        self.assertFalse(M.is_enabled(cfg, {"KEIBA_MAINTENANCE_WINDOW": "1"}))

    def test_clock_override_only_in_dev(self):
        env = {"KEIBA_ENV": "dev", "KEIBA_MAINTENANCE_NOW": "2026-10-21 03:10"}
        self.assertEqual(M.now_for_request(env).hour, 3)
        self.assertIsNone(M.now_for_request({"KEIBA_ENV": "stg", "KEIBA_MAINTENANCE_NOW": "2026-10-21 03:10"}))
        self.assertIsNone(M.now_for_request({"KEIBA_ENV": "prod"}, at="2026-10-21 03:10"))
        self.assertEqual(M.now_for_request({"KEIBA_ENV": "dev"}, at="2026-10-21 02:40").minute, 40)

    def test_load_config_hot_reload_and_broken_file(self):
        import tempfile, os

        d = tempfile.mkdtemp()
        p = os.path.join(d, "m.json")
        open(p, "w", encoding="utf-8").write(json.dumps({"daily": {"start": "01:00", "end": "02:00"}}))
        clock = [0.0]
        M.clear_cache()
        c1 = M.load_config(p, mono=lambda: clock[0])
        self.assertEqual(c1["daily"]["start"], "01:00")
        open(p, "w", encoding="utf-8").write(json.dumps({"daily": {"start": "05:00", "end": "06:00"}}))
        clock[0] = 10.0  # 30 秒未満はキャッシュ
        self.assertEqual(M.load_config(p, mono=lambda: clock[0])["daily"]["start"], "01:00")
        clock[0] = 31.0
        self.assertEqual(M.load_config(p, mono=lambda: clock[0])["daily"]["start"], "05:00")
        open(p, "w", encoding="utf-8").write("{broken")
        clock[0] = 100.0
        self.assertEqual(M.load_config(p, mono=lambda: clock[0])["daily"]["start"], "03:00")  # 壊れたら既定値
        M.clear_cache()

    def test_shipped_config_is_valid(self):
        cfg = M.load_config(M.CONFIG_PATH, ttl=0)
        self.assertEqual(cfg["daily"], {"start": "03:00", "end": "04:00"})
        self.assertEqual(cfg["timezone"], "Asia/Tokyo")
        M.clear_cache()


class TestExemptAndBypass(unittest.TestCase):
    def test_exempt(self):
        for p in ("/api/health", "/api/maintenance/status", "/api/scrape-jobs", "/api/scrape-queue/run", "/api/auth/status",
                  "/login", "/static/x.css", "/data/image/icon.jpg", "/_next/static/a.js"):
            self.assertTrue(M.is_exempt(p, CFG), p)
        for p in ("/api/race/202604040201/predictions", "/api/v1/races", "/", "/weekly-predictions", "/api/betting/portfolio"):
            self.assertFalse(M.is_exempt(p, CFG), p)

    def test_internal_call(self):
        self.assertTrue(M.is_internal_call("127.0.0.1", {}))
        self.assertTrue(M.is_internal_call("::1", {"Host": "localhost:23220"}))
        self.assertTrue(M.is_internal_call("::ffff:127.0.0.1", {}))
        # Next.js のプロキシ経由（X-Forwarded-For 付き）は内部呼び出しではない
        self.assertFalse(M.is_internal_call("127.0.0.1", {"X-Forwarded-For": "203.0.113.5"}))
        self.assertFalse(M.is_internal_call("127.0.0.1", {"x-forwarded-host": "example.com"}))
        self.assertFalse(M.is_internal_call("203.0.113.5", {}))
        self.assertFalse(M.is_internal_call(None, {}))

    def test_bypass_token_and_admin(self):
        env = {"KEIBA_MAINTENANCE_BYPASS_TOKEN": "s3cret"}
        kw = dict(remote_addr="203.0.113.5", cfg=CFG, env=env)
        self.assertEqual(M.bypass_reason(headers={"X-Keiba-Maintenance-Bypass": "s3cret"}, is_admin=False, **kw), "token")
        self.assertIsNone(M.bypass_reason(headers={"X-Keiba-Maintenance-Bypass": "wrong"}, is_admin=False, **kw))
        self.assertIsNone(M.bypass_reason(headers={}, is_admin=False, **kw))
        self.assertEqual(M.bypass_reason(headers={}, is_admin=True, **kw), "admin")
        # トークン未設定なら、ヘッダーを付けても通らない（空文字同士の一致を許さない）
        self.assertIsNone(M.bypass_reason(headers={"X-Keiba-Maintenance-Bypass": ""}, is_admin=False, remote_addr="203.0.113.5", cfg=CFG, env={}))
        self.assertEqual(M.bypass_reason(headers={}, is_admin=False, remote_addr="127.0.0.1", cfg=CFG, env={}), "internal")


class TestResponses(unittest.TestCase):
    def test_block(self):
        st = M.compute_status(CFG, at(3, 20), enabled=True)
        p = M.block_payload(st)
        self.assertTrue(p["maintenance"])
        self.assertEqual(p["ends_at_label"], "04:00")
        h = M.block_headers(st)
        self.assertEqual(h["Retry-After"], str(40 * 60))
        self.assertIn("no-store", h["Cache-Control"])
        self.assertIn("04:00", M.block_html(st))

    def test_audit_transition(self):
        M._last_phase["phase"] = None
        normal = M.compute_status(CFG, at(1), enabled=True)
        notice = M.compute_status(CFG, at(2, 40), enabled=True)
        active = M.compute_status(CFG, at(3, 5), enabled=True)
        self.assertIsNone(M.audit_transition(normal))   # 初回は記録のみ
        self.assertEqual(M.audit_transition(notice), "notice")
        self.assertIsNone(M.audit_transition(notice))
        self.assertEqual(M.audit_transition(active), "start")
        self.assertEqual(M.audit_transition(normal), "end")
        M._last_phase["phase"] = None


if __name__ == "__main__":
    unittest.main()


class TestScrapeNotifySuppression(unittest.TestCase):
    """スクレイピング状況の通知は、メンテナンス時間帯（と終了後 10 分）を『予定された停止』として異常にしない。"""

    def test_in_maintenance(self):
        from src.scripts.data import scrape_status_notify as S

        env = {"KEIBA_MAINTENANCE_WINDOW": "1", "KEIBA_ENV": "stg"}
        self.assertFalse(S.in_maintenance(at(2, 59), env))
        self.assertTrue(S.in_maintenance(at(3, 30), env))
        self.assertTrue(S.in_maintenance(at(4, 5), env))     # 終了直後の猶予
        self.assertFalse(S.in_maintenance(at(4, 10), env))
        self.assertFalse(S.in_maintenance(at(3, 30), {"KEIBA_ENV": "stg"}))  # 機能が無効なら抑制しない

    def test_run_skips_during_window(self):
        from src.scripts.data import scrape_status_notify as S

        for mode in ("alerts", "summary", "coverage"):
            r = S.run(mode, at(3, 30), send=True, inputs={"maintenance": True}, env={"KEIBA_SLACK_NOTIFY": "1"})
            self.assertEqual(r, {"code": 0, "mode": mode, "reason": "maintenance"})


class TestRuntimeOverride(unittest.TestCase):
    def test_runtime_override_file_extends_window(self):
        import os, tempfile
        from unittest import mock

        d = tempfile.mkdtemp()
        ov = os.path.join(d, "ov.json")
        cfgp = os.path.join(d, "m.json")
        open(cfgp, "w", encoding="utf-8").write("{}")
        open(ov, "w", encoding="utf-8").write(json.dumps({"override": {"start": "2026-10-21 03:00", "end": "2026-10-21 04:40", "message": "反映のため延長"}}))
        with mock.patch.dict(os.environ, {"KEIBA_MAINTENANCE_OVERRIDE": ov}):
            M.clear_cache()
            cfg = M.load_config(cfgp, ttl=0)
            s0 = M.compute_status(cfg, at(3, 30), enabled=True)   # 毎日の窓と重なる間は、終了が遅い延長を優先
            self.assertEqual(s0["ends_at_label"], "04:40")
            s = M.compute_status(cfg, at(4, 20), enabled=True)
            self.assertTrue(s["active"])
            self.assertEqual(s["ends_at_label"], "04:40")
            self.assertEqual(s["message"], "反映のため延長")
        with mock.patch.dict(os.environ, {"KEIBA_MAINTENANCE_OVERRIDE": os.path.join(d, "none.json")}):
            M.clear_cache()
            self.assertFalse(M.compute_status(M.load_config(cfgp, ttl=0), at(4, 20), enabled=True)["active"])
        M.clear_cache()


class TestStartEndNotify(unittest.TestCase):
    def test_flags_and_state(self):
        import tempfile
        from pathlib import Path

        from src.scripts.ops import maintenance_notify as N

        env = {"KEIBA_MAINTENANCE_WINDOW": "1", "KEIBA_SLACK_NOTIFY": "1", "KEIBA_ENV": "stg"}
        on = M._merge(CFG, {"notify": {"slack_start_end": True}})
        sent = Path(tempfile.mkdtemp())
        calls = []
        notify = lambda t, l, **kw: calls.append((t, l, kw)) or {"ok": True}
        # 設定が false なら送らない
        self.assertIn("false", N.run("start", now=at(3, 0), send=True, env=env, cfg=CFG, sent_dir=sent, notify=notify)["reason"])
        # 状態が合わなければ送らない（start は窓の中、end は窓の外）
        self.assertIn("合わない", N.run("start", now=at(5, 0), send=True, env=env, cfg=on, sent_dir=sent, notify=notify)["reason"])
        self.assertIn("合わない", N.run("end", now=at(3, 30), send=True, env=env, cfg=on, sent_dir=sent, notify=notify)["reason"])
        # dry-run は送らない
        self.assertEqual(N.run("start", now=at(3, 0), send=False, env=env, cfg=on, sent_dir=sent, notify=notify)["reason"], "ok")
        self.assertEqual(calls, [])
        # 送る → 同じ日の同じイベントは 2 回送らない
        self.assertEqual(N.run("start", now=at(3, 0), send=True, env=env, cfg=on, sent_dir=sent, notify=notify)["code"], 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(N.run("start", now=at(3, 1), send=True, env=env, cfg=on, sent_dir=sent, notify=notify)["reason"], "already_sent")
        self.assertEqual(N.run("end", now=at(4, 1), send=True, env=env, cfg=on, sent_dir=sent, notify=notify)["code"], 0)
        self.assertEqual(len(calls), 2)
        # Slack 側が無効なら 13
        self.assertEqual(N.run("end", now=at(4, 1, ), send=True, env={**env, "KEIBA_SLACK_NOTIFY": "0"}, cfg=on, sent_dir=Path(tempfile.mkdtemp()), notify=notify)["code"], 13)
