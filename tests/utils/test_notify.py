"""notify_slack のユニットテスト。"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from src.utils.notify import notify_slack


class TestNotifySlack(unittest.TestCase):
    def test_returns_false_when_webhook_unset(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(notify_slack("test message"))

    def test_returns_true_on_successful_post(self):
        with patch.dict("os.environ", {"SLACK_WEBHOOK_URL": "https://example.com/webhook"}):
            with patch("src.utils.notify.requests.post") as mock_post:
                mock_post.return_value.raise_for_status.return_value = None
                self.assertTrue(notify_slack("test message"))
                mock_post.assert_called_once()
                args, kwargs = mock_post.call_args
                self.assertEqual(args[0], "https://example.com/webhook")
                self.assertEqual(kwargs["json"], {"text": "test message"})

    def test_returns_false_and_swallows_exception_on_failure(self):
        with patch.dict("os.environ", {"SLACK_WEBHOOK_URL": "https://example.com/webhook"}):
            with patch("src.utils.notify.requests.post", side_effect=ConnectionError("boom")):
                self.assertFalse(notify_slack("test message"))


if __name__ == "__main__":
    unittest.main()
