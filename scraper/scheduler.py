```python
"""
構造チェック スケジューラー

毎朝6時にページ構造チェックを自動実行する。

実行方法:
  1. 直接起動 (常駐プロセス):
     python -m scraper.scheduler

  2. cron (推奨):
     0 6 * * * cd /home/hirokiakataoka/project/myproject/keiba && python -m scraper.scheduler --once

  3. systemd timer:
     systemd サービスファイルを生成:
     python -m scraper.scheduler --install-systemd
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("scraper.scheduler")


# ---------------------------------------------------------------------------
# SmartRC スケジューリング設定
# ---------------------------------------------------------------------------
# netkeibaと同一IDプロトコルを使用:
#   - レースID: 12桁 (例: 202306040811)
#   - 馬ID    : 10桁 (例: 2019105678)
# SmartRCのURLはnetkeibaのIDをそのまま流用して生成する。
# ---------------------------------------------------------------------------

class SmartRCConfig:
    """SmartRC アクセス制御設定。"""

    # リクエスト間隔 (秒)
    REQUEST_INTERVAL_MIN: float = 2.0
    REQUEST_INTERVAL_MAX: float = 5.0

    # セッション毎の上限リクエスト数
    SESSION_REQUEST_LIMIT: int = 200

    # 1日あたりの上限リクエスト数
    DAILY_REQUEST_LIMIT: int = 1000

    # SESSION_REQUEST_LIMIT 到達時のクールダウン (秒)
    SESSION_COOLDOWN_SEC: int = 60

    # アクセスログ保存先
    ACCESS_LOG_PATH: Path = Path("data/meta/logs/smartrc_access.log")

    # SmartRC ベースURL
    BASE_URL: str = "https://www.smartrc.jp"

    @classmethod
    def race_url(cls, race_id: str) -> str:
        """
        レースID (12桁) から SmartRC レースページURLを生成する。

        netkeibaと同一IDプロトコルを前提とする。
        例: race_id="202306040811"
            -> https://www.smartrc.jp/race/202306040811/
        """
        if len(race_id) != 12 or not race_id.isdigit():
            raise ValueError(f"レースIDは12桁の数字である必要があります: {race_id!r}")
        return f"{cls.BASE_URL}/race/{race_id}/"

    @classmethod
    def horse_url(cls, horse_id: str) -> str:
        """
        馬ID (10桁) から SmartRC 馬ページURLを生成する。

        netkeibaと同一IDプロトコルを前提とする。
        例: horse_id="2019105678"
            -> https://www.smartrc.jp/horse/2019105678/
        """
        if len(horse_id) != 10 or not horse_id.isdigit():
            raise ValueError(f"馬IDは10桁の数字である必要があります: {horse_id!r}")
        return f"{cls.BASE_URL}/horse/{horse_id}/"

    @classmethod
    def odds_url(cls, race_id: str) -> str:
        """
        レースID (12桁) から SmartRC オッズページURLを生成する。

        netkeibaと同一IDプロトコルを前提とする。
        例: race_id="202306040811"
            -> https://www.smartrc.jp/odds/202306040811/
        """
        if len(race_id) != 12 or not race_id.isdigit():
            raise ValueError(f"レースIDは12桁の数字である必要があります: {race_id!r}")
        return f"{cls.BASE_URL}/odds/{race_id}/"


class SmartRCAccessLogger:
    """SmartRC アクセスログ記録クラス。

    タイムスタンプ・URL・ステータスコード・レスポンスタイムを
    data/meta/logs/smartrc_access.log に CSV 形式で記録する。
    """

    FIELDNAMES = ["timestamp", "url", "status_code", "response_time_sec"]

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or SmartRCConfig.ACCESS_LOG_PATH
        self._ensure_log_file()

    def _ensure_log_file(self) -> None:
        """ログファイルとディレクトリを初期化する。"""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()
            logger.info("SmartRC アクセスログ初期化: %s", self.log_path)

    def record(
        self,
        url: str,
        status_code: int,
        response_time_sec: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """1件のアクセスをログに追記する。

        Args:
            url: アクセスしたURL
            status_code: HTTPステータスコード
            response_time_sec: レスポンスタイム (秒)
            timestamp: タイムスタンプ (省略時は現在時刻)
        """
        ts = (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "timestamp": ts,
            "url": url,
            "status_code": status_code,
            "response_time_sec": f"{response_time_sec:.3f}",
        }
        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(row)
        logger.debug("アクセスログ記録: %s %s %.3fs", status_code, url, response_time_sec)


class SmartRCScheduler:
    """SmartRC スクレイピング用スケジューラー。

    - リクエスト間隔: 2〜5秒 (ランダム)
    - 指数バックオフリトライ
    - 429/503 即時停止
    - SESSION_REQUEST_LIMIT 到達時に60秒クールダウン
    - アクセスログ記録
    """

    def __init__(
        self,
        config: Optional[SmartRCConfig] = None,
        access_logger: Optional[SmartRCAccessLogger] = None,
    ):
        self.config = config or SmartRCConfig()
        self.access_logger = access_logger or SmartRCAccessLogger()

        # セッション毎のリクエストカウンター
        self._session_request_count: int = 0

        # 本日のリクエストカウンター
        self._daily_request_count: int = 0
        self._daily_reset_date: datetime = datetime.now().date()

        logger.info(
            "SmartRCScheduler 初期化完了 "
            "(interval=%.1f-%.1fs, session_limit=%d, daily_limit=%d)",
            SmartRCConfig.REQUEST_INTERVAL_MIN,
            SmartRCConfig.REQUEST_INTERVAL_MAX,
            SmartRCConfig.SESSION_REQUEST_LIMIT,
            SmartRCConfig.DAILY_REQUEST_LIMIT,
        )

    # ------------------------------------------------------------------
    # カウンター管理
    # ------------------------------------------------------------------

    def _reset_daily_counter_if_needed(self) -> None:
        """日付が変わっていたら日次カウンターをリセットする。"""
        today = datetime.now().date()
        if today != self._daily_reset_date:
            logger.info(
                "日次リクエストカウンターをリセット (%s -> %s)",
                self._daily_reset_date,
                today,
            )
            self._daily_request_count = 0
            self._daily_reset_date = today

    def reset_session_counter(self) -> None:
        """セッションカウンターを手動リセットする。"""
        logger.info(
            "セッションカウンターをリセット (旧値: %d)", self._session_request_count
        )
        self._session_request_count = 0

    @property
    def session_request_count(self) -> int:
        return self._session_request_count

    @property
    def daily_request_count(self) -> int:
        self._reset_daily_counter_if_needed()
        return self._daily_request_count

    # ------------------------------------------------------------------
    # クールダウン / スロットリング
    # ------------------------------------------------------------------

    def _check_session_limit_and_cooldown(self) -> None:
        """SESSION_REQUEST_LIMIT に達していたら60秒クールダウンを挿入する。"""
        if self._session_request_count >= SmartRCConfig.SESSION_REQUEST_LIMIT:
            logger.warning(
                "セッションリクエスト上限 (%d) に達しました。%d秒クールダウンします。",
                SmartRCConfig.SESSION_REQUEST_LIMIT,
                SmartRCConfig.SESSION_COOLDOWN_SEC,
            )
            time.sleep(SmartRCConfig.SESSION_COOLDOWN_SEC)
            self._session_request_count = 0
            logger.info("クールダウン完了。セッションカウンターをリセットしました。")

    def _check_daily_limit(self) -> None:
        """DAILY_REQUEST_LIMIT に達していたら RuntimeError を送出する。"""
        self._reset_daily_counter_if_needed()
        if self._daily_request_count >= SmartRCConfig.DAILY_REQUEST_LIMIT:
            raise RuntimeError(
                f"本日の上限リクエスト数 ({SmartRCConfig.DAILY_REQUEST_LIMIT}) に達しました。"
                " 翌日まで待機してください。"
            )

    def _wait_interval(self) -> None:
        """REQUEST_INTERVAL_MIN〜MAX のランダム間隔だけ待機する。"""
        import random

        interval = random.uniform(
            SmartRCConfig.REQUEST_INTERVAL_MIN,
            SmartRCConfig.REQUEST_INTERVAL_MAX,
        )
        logger.debug("リクエスト間隔待機: %.2f秒", interval)
        time.sleep(interval)

    # ------------------------------------------------------------------
    # リクエスト実行
    # ------------------------------------------------------------------

    def fetch(
        self,
        url: str,
        max_retries: int = 3,
        backoff_base: float = 2.0,
    ) -> "requests.Response":  # type: ignore[name-defined]
        """
        URLをフェッチする。

        - 429 / 503 は即時停止 (RuntimeError)
        - その他エラーは指数バックオフでリトライ
        - アクセスログに記録する

        Args:
            url: フェッチするURL
            max_retries: 最大リトライ回数
            backoff_base: 指数バックオフの底 (秒)

        Returns:
            requests.Response オブジェクト

        Raises:
            RuntimeError: 429/503 受信時、または日次上限超過時
            requests.RequestException: リトライ上限超過時
        """
        try:
            import requests
        except ImportError as exc:
            raise ImportError("requests ライブラリが必要です: pip install requests") from exc

        self._check_daily_limit()
        self._check_session_limit_and_cooldown()

        last_exc: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                wait = backoff_base ** attempt
                logger.info(
                    "リトライ %d/%d — %.1f秒待機 (%s)",
                    attempt,
                    max_retries,
                    wait,
                    url,
                )
                time.sleep(wait)

            start_time = time.monotonic()
            try:
                response = requests.get(url, timeout=30)
                elapsed = time.monotonic() - start_time

                # アクセスログ記録
                self.access_logger.record(
                    url=url,
                    status_code=response.status_code,
                    response_time_sec=elapsed,
                )

                # カウンターインクリメント
                self._session_request_count += 1
                self._daily_request_count += 1

                # 429 / 503 は即時停止
                if response.status_code in (429, 503):
                    raise RuntimeError(
                        f"SmartRC から {response.status_code} が返されました。"
                        f" アクセスを即時停止します。URL: {url}"
                    )

                response.raise_for_status()

                # 次リクエストまでの間隔待機
                self._wait_interval()

                return response

            except RuntimeError:
                raise
            except Exception as exc:  # noqa: BLE001
                elapsed = time.monotonic() - start_time
                # エラー時もログ記録 (status_code=0)
                self.access_logger.record(
                    url=url,
                    status_code=0,
                    response_time_sec=elapsed,
                )
                last_exc = exc
                logger.warning("リクエスト失敗 (attempt %d): %s — %s", attempt, url, exc)

        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # URL生成ヘルパー (netkeibaと同一IDプロトコル)
    # ------------------------------------------------------------------

    @staticmethod
    def race_url(race_id: str) -> str:
        """レースID (12桁) から SmartRC URLを生成する。"""
        return SmartRCConfig.race_url(race_id)

    @staticmethod
    def horse_url(horse_id: str) -> str:
        """馬ID (10桁) から SmartRC URLを生成する。"""
        return SmartRCConfig.horse_url(horse_id)

    @staticmethod
    def odds_url(race_id: str) -> str:
        """レースID (12桁) から SmartRC オッズURLを生成する。"""
        return Sm
