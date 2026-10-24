"""jra_baba_live.py の構造変更検知（Slack通知）に関するunittest。

JRA公式ページへの実アクセスは行わず、BeautifulSoupでのパース対象HTMLを
モックで差し替え、"HTML構造が変わってセレクタがマッチしなくなった" ケースを
再現して検知ロジックを検証する。

対象TODO: docs/git_management/todo/cushion.md
「JRA公式ページの構造変更でライブ取得が失敗した場合の検知・アラートを追加する」
"""
from __future__ import annotations

import datetime
import unittest
from unittest.mock import patch

from src.scraper.jra_baba_live import JRABabaLiveScraper, run_cron_job


class _FakeResponse:
    """requests.Response の最小限の互換モック。"""

    def __init__(self, text: str, apparent_encoding: str = "utf-8"):
        self.text = text
        self.apparent_encoding = apparent_encoding
        self.encoding = None
        self.content = text.encode("utf-8")

    def raise_for_status(self) -> None:
        return None


class TestFetchCushionDataStructureAlert(unittest.TestCase):
    """_fetch_cushion_data: .unit はあるが .time/.cushion が1件も取れないケース。"""

    def test_unit_level_mismatch_reports_anomaly(self):
        html = """
        <html><body>
          <div id="rc01" title="東京">
            <div class="unit"><span class="broken_time">1月1日(土)</span></div>
            <div class="unit"><span class="broken_time">1月2日(日)</span></div>
          </div>
        </body></html>
        """
        scraper = JRABabaLiveScraper(output_dir="/tmp/_jra_baba_live_test_unused")
        with patch.object(scraper.session, "get", return_value=_FakeResponse(html)):
            result = scraper._fetch_cushion_data()

        self.assertEqual(result, {})
        self.assertEqual(len(scraper._structure_anomalies), 1)
        self.assertIn("東京", scraper._structure_anomalies[0])
        self.assertIn(".unit", scraper._structure_anomalies[0])

    def test_normal_structure_reports_no_anomaly(self):
        html = """
        <html><body>
          <div id="rc01" title="東京">
            <div class="unit">
              <span class="time">1月1日(土)</span>
              <span class="cushion">9.5</span>
            </div>
          </div>
        </body></html>
        """
        scraper = JRABabaLiveScraper(output_dir="/tmp/_jra_baba_live_test_unused")
        with patch.object(scraper.session, "get", return_value=_FakeResponse(html)):
            result = scraper._fetch_cushion_data()

        self.assertEqual(result, {"東京": [{"time": "1月1日(土)", "cushion": 9.5}]})
        self.assertEqual(scraper._structure_anomalies, [])

    def test_no_meeting_this_week_reports_no_anomaly(self):
        """開催なし週 (rc要素自体が無い) は構造変更の疑いとして報告しない。"""
        html = "<html><body><div class='no_meeting'>開催なし</div></body></html>"
        scraper = JRABabaLiveScraper(output_dir="/tmp/_jra_baba_live_test_unused")
        with patch.object(scraper.session, "get", return_value=_FakeResponse(html)):
            result = scraper._fetch_cushion_data()

        self.assertEqual(result, {})
        self.assertEqual(scraper._structure_anomalies, [])


class TestFetchVenueInfoStructureAlert(unittest.TestCase):
    """_fetch_venue_info: 見出しは取れるが想定フォーマットに一致しないケース。"""

    def test_header_format_mismatch_reports_anomaly(self):
        html = """
        <html><body>
          <div class="contents_header"><h2>想定外の見出しフォーマット</h2></div>
        </body></html>
        """
        scraper = JRABabaLiveScraper(output_dir="/tmp/_jra_baba_live_test_unused")
        with patch.object(scraper.session, "get", return_value=_FakeResponse(html)):
            result = scraper._fetch_venue_info()

        self.assertEqual(result, {})
        self.assertEqual(len(scraper._structure_anomalies), 1)
        self.assertIn("contents_header", scraper._structure_anomalies[0])

    def test_header_format_match_reports_no_anomaly(self):
        html = """
        <html><body>
          <div class="contents_header">
            <h2>第1回東京競馬第1日2026年1月1日（土）</h2>
          </div>
        </body></html>
        """
        scraper = JRABabaLiveScraper(output_dir="/tmp/_jra_baba_live_test_unused")
        with patch.object(scraper.session, "get", return_value=_FakeResponse(html)):
            result = scraper._fetch_venue_info()

        self.assertIn("東京", result)
        self.assertEqual(scraper._structure_anomalies, [])


class TestScrapeNotifiesOnAnomaly(unittest.TestCase):
    """scrape() が構造変更検知時に notify_slack を1回だけ呼ぶことを確認。"""

    def test_scrape_sends_single_slack_notification_on_anomaly(self):
        cushion_html = """
        <html><body>
          <div id="rc01" title="東京">
            <div class="unit"><span class="broken_time">1月1日(土)</span></div>
          </div>
        </body></html>
        """
        scraper = JRABabaLiveScraper(output_dir="/tmp/_jra_baba_live_test_unused")
        with patch.object(scraper.session, "get", return_value=_FakeResponse(cushion_html)), \
             patch("src.utils.notify.notify_slack") as mock_notify:
            records = scraper.scrape()

        self.assertEqual(records, [])
        mock_notify.assert_called_once()
        (sent_message,), _ = mock_notify.call_args
        self.assertIn("構造変更の疑い", sent_message)
        self.assertIn("東京", sent_message)

    def test_scrape_does_not_notify_when_no_meeting(self):
        html = "<html><body>開催なし</body></html>"
        scraper = JRABabaLiveScraper(output_dir="/tmp/_jra_baba_live_test_unused")
        with patch.object(scraper.session, "get", return_value=_FakeResponse(html)), \
             patch("src.utils.notify.notify_slack") as mock_notify:
            records = scraper.scrape()

        self.assertEqual(records, [])
        mock_notify.assert_not_called()


class TestRunCronJobStructureAlert(unittest.TestCase):
    """run_cron_job: 更新検知後のフルスクレイプが0件のときにSlack通知する。"""

    def test_notifies_when_full_scrape_returns_zero_after_hash_change(self):
        today = datetime.date.today().isoformat()
        fake_schedule = [{
            "date": today,
            "type": "race_day",
            "venues": ["東京"],
        }]

        with patch("src.scraper.jra_baba_live._load_poll_schedule", return_value=fake_schedule), \
             patch("src.scraper.jra_baba_live._in_any_window", return_value=True), \
             patch.object(JRABabaLiveScraper, "has_new_data", return_value=True), \
             patch.object(JRABabaLiveScraper, "scrape", return_value=[]), \
             patch("src.utils.notify.notify_slack") as mock_notify:
            count = run_cron_job()

        self.assertEqual(count, 0)
        mock_notify.assert_called_once()
        (sent_message,), _ = mock_notify.call_args
        self.assertIn("構造が変わり", sent_message)

    def test_does_not_notify_when_full_scrape_returns_records(self):
        today = datetime.date.today().isoformat()
        fake_schedule = [{
            "date": today,
            "type": "race_day",
            "venues": ["東京"],
        }]
        fake_records = [{"date": today, "venue_name": "東京", "venue_code": "05"}]

        with patch("src.scraper.jra_baba_live._load_poll_schedule", return_value=fake_schedule), \
             patch("src.scraper.jra_baba_live._in_any_window", return_value=True), \
             patch.object(JRABabaLiveScraper, "has_new_data", return_value=True), \
             patch.object(JRABabaLiveScraper, "scrape", return_value=fake_records), \
             patch("src.utils.notify.notify_slack") as mock_notify:
            count = run_cron_job()

        self.assertEqual(count, 1)
        mock_notify.assert_not_called()

    def test_does_not_notify_when_no_new_data(self):
        today = datetime.date.today().isoformat()
        fake_schedule = [{
            "date": today,
            "type": "race_day",
            "venues": ["東京"],
        }]

        with patch("src.scraper.jra_baba_live._load_poll_schedule", return_value=fake_schedule), \
             patch("src.scraper.jra_baba_live._in_any_window", return_value=True), \
             patch.object(JRABabaLiveScraper, "has_new_data", return_value=False), \
             patch("src.utils.notify.notify_slack") as mock_notify:
            count = run_cron_job()

        self.assertEqual(count, 0)
        mock_notify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
