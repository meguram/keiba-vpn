"""遅延ログイン（最初の通信中にログイン → 会員確認の通信）でロックが再入しても固まらないことのテスト。"""

from __future__ import annotations

import threading
import unittest
from unittest import mock

from src.scraper.client import NetkeibaClient


class _Resp:
    status_code = 200
    text = "<html>ok</html>"
    content = b"<html>ok</html>"
    headers: dict = {}
    encoding = "utf-8"
    apparent_encoding = "utf-8"


class LazyLoginTest(unittest.TestCase):
    def test_first_fetch_with_pending_login_does_not_deadlock(self):
        c = NetkeibaClient(auto_login=False)
        c._login_pending = True
        calls = []

        def fake_login(self_):
            self_._login_pending = False
            calls.append("login")
            self_.fetch("https://race.netkeiba.com/race/speed.html?race_id=1", use_cache=False)       # 会員確認の通信（外側の fetch が持つロックに再入する）
            return True

        def fake_backoff(self_, url):
            _ = self_._session                       # ここで保留中のログインが走る
            calls.append(url)
            return _Resp()

        done = threading.Event()

        def run():
            with mock.patch.object(NetkeibaClient, "login", fake_login), mock.patch.object(NetkeibaClient, "_get_with_backoff", fake_backoff), \
                    mock.patch.object(NetkeibaClient, "_check_burst", lambda s: None), mock.patch.object(NetkeibaClient, "_increment_request_count", lambda s, u: None):
                c.fetch("https://race.netkeiba.com/top/x.html", use_cache=False)
            done.set()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        self.assertTrue(done.wait(5), "最初の通信で固まった（ロックの再入ができていない）")
        self.assertIn("https://race.netkeiba.com/top/x.html", calls)
        self.assertIn("login", calls)


if __name__ == "__main__":
    unittest.main()


class SessionRefreshReloginTest(unittest.TestCase):
    def test_login_state_is_reset_so_login_really_runs_after_refresh(self):
        import src.scraper.client as cl

        c = NetkeibaClient(auto_login=False)
        c._logged_in = True
        c._request_count = cl._SESSION_REFRESH_INTERVAL
        seen = {}

        def fake_login(self_):
            seen["logged_in_at_login"] = self_._logged_in      # True のままだと本物の login() は早期 return して入り直さない
            return True

        with mock.patch.object(NetkeibaClient, "login", fake_login), mock.patch.object(NetkeibaClient, "_refresh_session", lambda s: None), \
                mock.patch("time.sleep", lambda *_: None):
            c._check_burst()
        self.assertEqual(seen.get("logged_in_at_login"), False)
