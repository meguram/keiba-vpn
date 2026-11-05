"""ログイン確認ページと将来日の一覧の 400 は、ブロックとして数えず即座に諦めることのテスト。"""

from __future__ import annotations

import unittest
from unittest import mock

import requests

from src.scraper import client as cl

PROBE = f"https://race.netkeiba.com/race/speed.html?race_id={cl._login_probe_race_id()}"
FUTURE = "https://race.netkeiba.com/top/race_list_get_date_list.html?kaisai_date=29991231&encoding=UTF-8"
DB_FUTURE = "https://db.netkeiba.com/race/list/29991231/"
DB_PAST = "https://db.netkeiba.com/race/list/20200105/"
PAST = "https://race.netkeiba.com/top/race_list_get_date_list.html?kaisai_date=20200105&encoding=UTF-8"
OTHER_SPEED = "https://race.netkeiba.com/race/speed.html?race_id=202509040411"


class _Resp:
    status_code = 400
    text = ""
    content = b""
    headers: dict = {}

    def raise_for_status(self):
        raise requests.HTTPError("400 Client Error", response=self)


class Soft400Test(unittest.TestCase):
    def test_classification(self):
        self.assertTrue(cl._is_soft_400_url(PROBE))
        self.assertTrue(cl._is_soft_400_url(FUTURE))
        self.assertTrue(cl._is_soft_400_url(DB_FUTURE))
        self.assertFalse(cl._is_soft_400_url(DB_PAST))
        self.assertFalse(cl._is_soft_400_url(PAST))
        self.assertFalse(cl._is_soft_400_url(OTHER_SPEED))

    def test_soft_url_gives_up_without_counting(self):
        c = cl.NetkeibaClient(auto_login=False)
        seen = []
        for url in (PROBE, FUTURE, DB_FUTURE):
            with mock.patch.object(c._session, "get", return_value=_Resp()) as g, \
                    mock.patch.object(cl, "_RESPONSE_OBSERVERS", [lambda u, r: seen.append(u)]), \
                    mock.patch("src.scraper.http400_strategy.global_400_rate_monitor") as mon:
                with self.assertRaises(requests.HTTPError):
                    c._get_with_backoff(url)
                self.assertEqual(g.call_count, 1)
                mon.assert_not_called()
        self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()
