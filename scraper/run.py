"""
スクレイピング CLIランナー

Usage:
  python -m scraper.run race-result 202505020811
  python -m scraper.run race-card 202606020611
  python -m scraper.run horse 2021105354
  python -m scraper.run race-list 20260322
  python -m scraper.run race-detail 202606020611
  python -m scraper.run summary
  python -m scraper.run reparse race_shutuba              # 全 race_shutuba を再パース
  python -m scraper.run reparse-one race_result 202505020811  # 1件だけ再パース

ストレージ:
  GCS をプライマリとする HybridStorage (JSON) を使用。
  生 HTML は gzip 圧縮で HtmlArchive に GCS 保全。
  パーサー更新時は reparse で JSON を再生成可能。

データフロー:
  fetch(url) → HTML
    ├── HtmlArchive.save(category, key, html)  → gzip → GCS
    └── parser.parse(html) → dict
          └── HybridStorage.save(category, key, dict) → JSON → GCS
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import random
import sys
import threading
import time
from datetime import datetime, timedelta

import requests

from scraper.client import NetkeibaClient
from scraper.parsers import (
    RaceResultParser, RaceCardParser, HorseParser, RaceListParser,
    SpeedIndexParser, ShutubaPastParser, OddsParser,
    PaddockParser, BarometerParser, OikiriParser, TrainingParser,
    TrainerCommentParser,
)
from scraper.storage import HybridStorage
from scraper.html_archive import HtmlArchive

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("scraper.run")


class ScraperRunner:
    """各スクレイピングタスクを統合するランナー (GCS プライマリ + HTML アーカイブ)"""

    RACE_RESULT_URL = "https://db.netkeiba.com/race/{race_id}/"
    RACE_CARD_URL = "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    SPEED_INDEX_URL = "https://race.netkeiba.com/race/speed.html?race_id={race_id}"
    SHUTUBA_PAST_URL = "https://race.netkeiba.com/race/shutuba_past.html?race_id={race_id}"
    HORSE_URL = "https://db.netkeiba.com/horse/{horse_id}/"
    HORSE_RESULT_URL = "https://db.netkeiba.com/horse/result/{horse_id}/"
    HORSE_PED_URL = "https://db.netkeiba.com/horse/ped/{horse_id}/"
    RACE_LIST_DB_URL = "https://db.netkeiba.com/race/list/{date}/"
    ODDS_URL = "https://race.netkeiba.com/odds/index.html?type=b1&race_id={race_id}"
    PADDOCK_URL = "https://race.netkeiba.com/race/paddock.html?race_id={race_id}"
    BAROMETER_URL = "https://db.netkeiba.com/race/barometer/{race_id}/"
    OIKIRI_URL = "https://race.netkeiba.com/race/oikiri.html?race_id={race_id}"
    TRAINER_COMMENT_URL = "https://race.netkeiba.com/race/comment.html?race_id={race_id}"
    HORSE_TRAINING_URL = "https://db.netkeiba.com/horse/training.html?id={horse_id}"

    def __init__(self, interval: float = 1.0, cache: bool = True,
                 auto_login: bool = True):
        self.client = NetkeibaClient(
            interval=interval,
            cache_dir=None,
            auto_login=auto_login,
        )
        self.storage = HybridStorage(".")
        self.archive = HtmlArchive()

        self.result_parser = RaceResultParser()
        self.card_parser = RaceCardParser()
        self.horse_parser = HorseParser()
        self.list_parser = RaceListParser()
        self.speed_parser = SpeedIndexParser()
        self.past_parser = ShutubaPastParser()
        self.odds_parser = OddsParser()
        self.paddock_parser = PaddockParser()
        self.barometer_parser = BarometerParser()
        self.oikiri_parser = OikiriParser()
        self.trainer_comment_parser = TrainerCommentParser()
        self.training_parser = TrainingParser()

        self._smartrc_client = None
        self._smartrc_checked = False

        self._horse_freshness_days = 7

        self._horse_session_cache: dict[str, float] = {}
        self._horse_blob_prefetched: set[str] = set()
        self._horse_cache_lock = threading.Lock()

    # ── 馬情報の鮮度判定 ──────────────────────────────

    def prefetch_horse_timestamps(self, year: str = "") -> None:
        """
        horse_result のタイムスタンプを batch_list_blobs で一括取得し、
        セッションキャッシュに格納する。
        大規模バッチの冒頭で呼ぶと、以降の is_horse_fresh が GCS 0 回で動作する。
        """
        if not year:
            year = datetime.now().strftime("%Y")
        if year in self._horse_blob_prefetched:
            return
        blobs = self.storage.batch_list_blobs("horse_result", year)
        self._horse_session_cache.update(blobs)
        self._horse_blob_prefetched.add(year)
        logger.info("馬情報タイムスタンプをプリフェッチ: year=%s, %d件", year, len(blobs))

    def mark_horse_scraped(self, horse_id: str) -> None:
        """セッション内で取得済みとして記録する。"""
        with self._horse_cache_lock:
            self._horse_session_cache[horse_id] = time.time()

    def is_horse_fresh(self, horse_id: str, race_date: str = "") -> bool:
        """
        馬情報が再取得不要かを判定する。

        判定基準:
          1. GCS に horse_result/{horse_id} が存在する
          2. 取得日 (scraped_at) がレース開催日以降である
          3. 取得日から self._horse_freshness_days (デフォルト7日) 以内である

        セッションキャッシュがある場合は GCS コールなしで判定可能。
        スレッドセーフ: _horse_cache_lock でキャッシュアクセスを保護。
        """
        now = time.time()
        threshold = self._horse_freshness_days * 86400

        with self._horse_cache_lock:
            cached_time = self._horse_session_cache.get(horse_id)
        if cached_time is not None:
            if (now - cached_time) <= threshold:
                if race_date and len(race_date) == 8:
                    try:
                        race_dt = datetime.strptime(race_date, "%Y%m%d")
                        if datetime.fromtimestamp(cached_time).date() < race_dt.date():
                            return False
                    except (ValueError, OSError):
                        pass
                return True

        data = self.storage.load("horse_result", horse_id)
        if data is None:
            return False

        scraped_at = float(data.get("_meta", {}).get("scraped_at", 0))
        if scraped_at == 0:
            scraped_at = self.storage.get_scraped_at_from_blob("horse_result", horse_id)
        if scraped_at == 0:
            return False

        with self._horse_cache_lock:
            self._horse_session_cache[horse_id] = scraped_at

        if (now - scraped_at) > threshold:
            return False

        if race_date and len(race_date) == 8:
            try:
                race_dt = datetime.strptime(race_date, "%Y%m%d")
                if datetime.fromtimestamp(scraped_at).date() < race_dt.date():
                    return False
            except (ValueError, OSError):
                pass

        return True

    @property
    def smartrc(self):
        """SmartRC クライアント (遅延初期化)。"""
        if not self._smartrc_checked:
            self._init_smartrc()
        return self._smartrc_client

    # ── 汎用: fetch → archive → parse → save ────────────

    def _fetch_parse_save(
        self,
        category: str,
        key: str,
        url: str,
        parser,
        parse_kwargs: dict | None = None,
        skip_existing: bool = True,
        need_login: bool = False,
        min_entries: int = 0,
    ) -> dict | None:
        """
        共通フロー: fetch HTML → archive → parse → save JSON。

        Args:
            category: GCS カテゴリ名 (race_result, race_shutuba, ...)
            key: race_id / horse_id
            url: スクレイピング対象 URL
            parser: parse(html, **kwargs) を持つパーサー
            parse_kwargs: parser.parse に渡す追加引数
            skip_existing: 既存 JSON があればスキップ
            need_login: ログインが必要か
            min_entries: entries の最小数 (0 ならチェックしない)
        """
        if skip_existing and self.storage.exists(category, key):
            logger.info("スキップ (既存): %s/%s", category, key)
            return self.storage.load(category, key)

        if need_login and not self.client.is_logged_in:
            logger.info("%s の取得にログインが必要 — ログインを試行します", category)
            if not self.client.login():
                logger.warning("%s の取得にはログインが必要ですが、ログインに失敗しました", category)
                return None

        try:
            html = self.client.fetch(url)

            self.archive.save(category, key, html)

            kwargs = parse_kwargs or {}
            data = parser.parse(html, **kwargs)

            if min_entries > 0 and len(data.get("entries", [])) < min_entries:
                logger.warning("データ不足: %s/%s (%d entries)", category, key,
                               len(data.get("entries", [])))
                return None

            self.storage.save(category, key, data)
            n = len(data.get("entries", []))
            label = data.get("race_name", "") or data.get("horse_name", "") or key
            logger.info("取得完了: %s/%s — %s (%d件)", category, key, label, n)
            return data

        except Exception as e:
            logger.error("取得失敗 [%s/%s]: %s", category, key, e)
            return None

    # ── 個別スクレイピングメソッド ──────────────────────────

    def scrape_race_result(self, race_id: str, skip_existing: bool = True) -> dict | None:
        return self._fetch_parse_save(
            "race_result", race_id,
            self.RACE_RESULT_URL.format(race_id=race_id),
            self.result_parser, {"race_id": race_id},
            skip_existing=skip_existing, min_entries=1,
        )

    def scrape_race_card(self, race_id: str, skip_existing: bool = True) -> dict | None:
        return self._fetch_parse_save(
            "race_shutuba", race_id,
            self.RACE_CARD_URL.format(race_id=race_id),
            self.card_parser, {"race_id": race_id},
            skip_existing=skip_existing, min_entries=1,
        )

    def scrape_speed_index(self, race_id: str, skip_existing: bool = True) -> dict | None:
        return self._fetch_parse_save(
            "race_index", race_id,
            self.SPEED_INDEX_URL.format(race_id=race_id),
            self.speed_parser, {"race_id": race_id},
            skip_existing=skip_existing, need_login=True,
        )

    def scrape_shutuba_past(self, race_id: str, skip_existing: bool = True) -> dict | None:
        if skip_existing and self.storage.exists("race_shutuba_past", race_id):
            logger.info("スキップ (既存): race_shutuba_past/%s", race_id)
            return self.storage.load("race_shutuba_past", race_id)

        url = self.SHUTUBA_PAST_URL.format(race_id=race_id)
        try:
            html = self.client.fetch(url)
            self.archive.save("race_shutuba_past", race_id, html)

            data = self.past_parser.parse(html, race_id=race_id)
            n_entries = len(data.get("entries", []))
            n_train = len(data.get("training", []))
            if n_entries or n_train:
                self.storage.save("race_shutuba_past", race_id, data)
                logger.info("馬柱+調教取得: %s (%d頭, 調教%d件)", race_id, n_entries, n_train)
                return data
            else:
                logger.warning("馬柱データなし: %s", race_id)
                return None
        except Exception as e:
            logger.error("馬柱取得失敗 [%s]: %s", race_id, e)
            return None

    def scrape_odds(self, race_id: str, skip_existing: bool = True) -> dict | None:
        """オッズを JSON API から取得する (HTML は JS 動的レンダリングのため)。"""
        if skip_existing and self.storage.exists("race_odds", race_id):
            logger.info("スキップ (既存): race_odds/%s", race_id)
            return self.storage.load("race_odds", race_id)

        try:
            data = self.odds_parser.parse_from_api(
                self.client._session, race_id
            )
            if data.get("entries"):
                self.storage.save("race_odds", race_id, data)
                logger.info("保存: race_odds/%s (%d件)", race_id,
                            len(data["entries"]))
                return data
            else:
                logger.warning("オッズデータなし: %s", race_id)
                return None
        except Exception as e:
            logger.error("オッズ取得失敗 [%s]: %s", race_id, e)
            return None

    def scrape_pair_odds(self, race_id: str, skip_existing: bool = True) -> dict | None:
        """2連系オッズ (馬連/ワイド/馬単) を JSON API から取得する。"""
        if skip_existing and self.storage.exists("race_pair_odds", race_id):
            logger.info("スキップ (既存): race_pair_odds/%s", race_id)
            return self.storage.load("race_pair_odds", race_id)

        try:
            data = self.odds_parser.parse_pair_odds_from_api(
                self.client._session, race_id
            )
            n_umaren = len(data.get("umaren", []))
            n_wide = len(data.get("wide", []))
            n_umatan = len(data.get("umatan", []))
            if n_umaren or n_wide or n_umatan:
                self.storage.save("race_pair_odds", race_id, data)
                logger.info("保存: race_pair_odds/%s (馬連=%d, ワイド=%d, 馬単=%d)",
                            race_id, n_umaren, n_wide, n_umatan)
                return data
            else:
                logger.warning("2連系オッズデータなし: %s", race_id)
                return None
        except Exception as e:
            logger.error("2連系オッズ取得失敗 [%s]: %s", race_id, e)
            return None

    def scrape_paddock(self, race_id: str, skip_existing: bool = True) -> dict | None:
        return self._fetch_parse_save(
            "race_paddock", race_id,
            self.PADDOCK_URL.format(race_id=race_id),
            self.paddock_parser, {"race_id": race_id},
            skip_existing=skip_existing, need_login=True,
        )

    def scrape_barometer(self, race_id: str, skip_existing: bool = True) -> dict | None:
        """バロメーター (走行データ/タイム指数) を AJAX API から取得する。"""
        if skip_existing and self.storage.exists("race_barometer", race_id):
            logger.info("スキップ (既存): race_barometer/%s", race_id)
            return self.storage.load("race_barometer", race_id)

        if not self.client.is_logged_in:
            if not self.client.login():
                logger.warning("バロメーター取得にはログインが必要です")
                return None

        try:
            data = self.barometer_parser.parse_from_api(
                self.client._session, race_id
            )
            if data.get("entries"):
                self.storage.save("race_barometer", race_id, data)
                logger.info("保存: race_barometer/%s (%d件)", race_id,
                            len(data["entries"]))
                return data
            else:
                logger.warning("バロメーターデータなし: %s", race_id)
                return None
        except Exception as e:
            logger.error("バロメーター取得失敗 [%s]: %s", race_id, e)
            return None

    def scrape_oikiri(self, race_id: str, skip_existing: bool = True) -> dict | None:
        return self._fetch_parse_save(
            "race_oikiri", race_id,
            self.OIKIRI_URL.format(race_id=race_id),
            self.oikiri_parser, {"race_id": race_id},
            skip_existing=skip_existing,
        )

    def scrape_trainer_comment(self, race_id: str, skip_existing: bool = True) -> dict | None:
        return self._fetch_parse_save(
            "race_trainer_comment", race_id,
            self.TRAINER_COMMENT_URL.format(race_id=race_id),
            self.trainer_comment_parser, {"race_id": race_id},
            skip_existing=skip_existing, need_login=True,
        )

    # ── 馬情報 (2ページ分の HTML を個別にアーカイブ) ──────────

    def scrape_horse(self, horse_id: str, skip_existing: bool = True,
                     with_history: bool = True) -> dict | None:
        if skip_existing and self.storage.exists("horse_result", horse_id):
            logger.info("スキップ (既存): horse_result/%s", horse_id)
            return self.storage.load("horse_result", horse_id)

        url = self.HORSE_URL.format(horse_id=horse_id)
        try:
            html = self.client.fetch(url)
            self.archive.save("horse_profile", horse_id, html)

            result_html = None
            if with_history:
                result_url = self.HORSE_RESULT_URL.format(horse_id=horse_id)
                result_html = self.client.fetch(result_url)
                self.archive.save("horse_result_html", horse_id, result_html)

            ped_html = None
            try:
                ped_url = self.HORSE_PED_URL.format(horse_id=horse_id)
                ped_html = self.client.fetch(ped_url)
                self.archive.save("horse_ped", horse_id, ped_html)
            except Exception as e:
                logger.debug("血統ページ取得失敗 [%s]: %s", horse_id, e)

            data = self.horse_parser.parse(
                html, horse_id=horse_id,
                result_html=result_html, ped_html=ped_html,
            )
            if data.get("horse_name"):
                self.storage.save("horse_result", horse_id, data)
                self.mark_horse_scraped(horse_id)
                n_races = len(data.get("race_history", []))
                logger.info("取得完了: %s - %s (%d戦) sire=%s",
                            horse_id, data["horse_name"], n_races, data.get("sire", "?"))
                return data
            else:
                logger.warning("馬情報なし: %s", horse_id)
                return None
        except Exception as e:
            logger.error("取得失敗 [%s]: %s", horse_id, e)
            return None

    def scrape_horse_pedigree_5gen(
        self, horse_id: str, skip_existing: bool = False
    ) -> dict | None:
        """
        db.netkeiba 血統ページから 5 世代表をパースし horse_pedigree_5gen に保存する。
        キュータスク horse_pedigree_5gen から呼ばれる。
        """
        from research.pedigree_similarity import parse_blood_table_5gen
        from scripts.scrape_pedigree_5gen import build_pedigree_record

        if skip_existing:
            ex = self.storage.load("horse_pedigree_5gen", horse_id)
            anc = (ex or {}).get("ancestors") or []
            if ex and len(anc) >= 5:
                logger.info("スキップ (既存): horse_pedigree_5gen/%s", horse_id)
                return ex

        url = self.HORSE_PED_URL.format(horse_id=horse_id)
        html = self.client.fetch(url)
        self.archive.save("horse_ped", horse_id, html)
        ancestors = parse_blood_table_5gen(html)
        if len(ancestors) < 5:
            raise ValueError(f"5世代血統 祖先不足: {len(ancestors)}")
        rec = build_pedigree_record(
            horse_id, ancestors, source="queue_horse_pedigree_5gen"
        )
        self.storage.save("horse_pedigree_5gen", horse_id, rec)
        logger.info(
            "保存: horse_pedigree_5gen/%s (%d頭)", horse_id, len(ancestors)
        )
        return rec

    # ── 調教タイム (全ページ結合) ────────────────────────────

    def scrape_horse_training(
        self,
        horse_id: str,
        skip_existing: bool = False,
        max_pages: int = 20,
    ) -> dict | None:
        """
        馬の調教タイムを全ページ分スクレイピングし、結合して保存する。

        ページネーションは ?page=N で遷移（10件/ページ）。
        全ページを取得し、entries を時系列で結合する。

        鮮度管理: skip_existing=False（デフォルト）で常に最新化。
                  True の場合は既存データが7日以内なら再取得しない。
        """
        if skip_existing:
            existing = self.storage.load("horse_training", horse_id)
            if existing:
                scraped_at = existing.get("_meta", {}).get("scraped_at", 0)
                age_days = (time.time() - scraped_at) / 86400 if scraped_at else 999
                if age_days < self._horse_freshness_days:
                    logger.info("スキップ (鮮度OK): horse_training/%s (%.1f日前)",
                                horse_id, age_days)
                    return existing

        if not self.client.is_logged_in:
            if not self.client.login():
                logger.warning("調教データの取得にはログインが必要です")
                return None

        all_entries: list[dict] = []
        total_items = 0
        max_page = 1

        for page in range(1, max_pages + 1):
            url = self.HORSE_TRAINING_URL.format(horse_id=horse_id)
            if page > 1:
                url += f"&page={page}"

            try:
                html = self.client.fetch(url, use_cache=False)
            except Exception as e:
                logger.error("調教ページ取得失敗 [%s] page=%d: %s", horse_id, page, e)
                if page == 1:
                    return None
                break

            self.archive.save("horse_training", f"{horse_id}_p{page}", html)

            data = self.training_parser.parse(html, horse_id=horse_id)

            if page == 1:
                total_items = data.get("total_items", 0)
                max_page = data.get("max_page", 1)
                if total_items == 0 and not data.get("entries"):
                    logger.warning("調教データなし: %s", horse_id)
                    return None

            entries = data.get("entries", [])
            if not entries:
                break

            all_entries.extend(entries)
            logger.info("調教取得: %s page=%d/%d (%d件, 累計%d件)",
                        horse_id, page, max_page, len(entries), len(all_entries))

            if page >= max_page:
                break

            time.sleep(random.uniform(1.0, 2.0))

        seen = set()
        unique_entries = []
        for e in all_entries:
            key = (e["date"], e["course"], e["time_raw"])
            if key not in seen:
                seen.add(key)
                unique_entries.append(e)

        result = {
            "horse_id": horse_id,
            "total_items": total_items,
            "pages_fetched": min(max_page, max_pages),
            "entries": unique_entries,
        }
        self.storage.save("horse_training", horse_id, result)
        logger.info("調教保存完了: %s — %d件 (%dページ)",
                     horse_id, len(unique_entries), result["pages_fetched"])
        return result

    # ── レース詳細 (マージ) ──────────────────────────────

    def scrape_race_detail(self, race_id: str) -> dict:
        card = self.scrape_race_card(race_id, skip_existing=False)
        speed = self.scrape_speed_index(race_id, skip_existing=False)
        past = self.scrape_shutuba_past(race_id, skip_existing=False)

        if not card:
            return {}

        speed_map = {}
        if speed:
            for e in speed.get("entries", []):
                speed_map[e["horse_number"]] = {
                    "speed_max": e.get("speed_max", 0),
                    "speed_avg": e.get("speed_avg", 0),
                    "speed_distance": e.get("speed_distance", 0),
                    "speed_course": e.get("speed_course", 0),
                    "speed_recent": e.get("speed_recent", []),
                }

        past_map = {}
        if past:
            for e in past.get("entries", []):
                past_map[e["horse_number"]] = e.get("past_races", [])

        for entry in card.get("entries", []):
            hn = entry["horse_number"]
            if hn in speed_map:
                entry.update(speed_map[hn])
            if hn in past_map:
                entry["past_races"] = past_map[hn]

        self.storage.save("race_detail", race_id, card)
        logger.info("レース詳細取得完了: %s (%d頭)", race_id, len(card.get("entries", [])))
        return card

    # ── レース一覧 ──────────────────────────────────────

    def scrape_race_list(self, date: str) -> list[dict]:
        """
        日付別レース一覧を保存する。

        - JST で「直近」開催日: race.netkeiba.com/top が JS 経由で取る断片 HTML と同じ URL を
          先に叩く（db.netkeiba の /race/list/ はこの期間ほぼ空になりがち）。
        - それ以外・またはトップが空: db.netkeiba を従来どおり。
        - トップ優先でも空なら db にフォールバック、db だけの日で空ならトップにフォールバック。
        """
        from scraper.netkeiba_top_race_list import (
            fetch_races_for_kaisai_date,
            kaisai_date_in_top_priority_window,
        )

        races: list[dict] = []
        list_meta: dict = {}
        sub_html: str | None = None

        def _save_and_return(rlist: list[dict], source: str) -> list[dict]:
            payload: dict = {"date": date, "races": rlist}
            if source:
                payload["_meta"] = {"race_list_source": source}
            self.storage.save("race_lists", date, payload)
            logger.info("レース一覧取得: %s - %d レース (%s)", date, len(rlist), source)
            return rlist

        prefer_top = kaisai_date_in_top_priority_window(date)
        if prefer_top:
            try:
                top_r, raw_sub = fetch_races_for_kaisai_date(
                    self.client,
                    date,
                    use_cache=False,
                    return_sub_html=True,
                )
                if top_r:
                    races = top_r
                    sub_html = raw_sub or None
                    list_meta["race_list_source"] = "race.netkeiba.com/top"
            except Exception as e:
                logger.warning("レース一覧(netkeiba top) 失敗 [%s]: %s", date, e)

        if not races:
            url = self.RACE_LIST_DB_URL.format(date=date)
            try:
                html = self.client.fetch(url, use_cache=False)
                self.archive.save("race_lists", date, html)
                data = self.list_parser.parse(html, date=date)
                races = data.get("races", [])
                if races:
                    list_meta["race_list_source"] = "db.netkeiba.com"
            except Exception as e:
                logger.error("レース一覧(db) 失敗 [%s]: %s", date, e)
                races = []

        if not races and not prefer_top:
            try:
                top_r, raw_sub = fetch_races_for_kaisai_date(
                    self.client,
                    date,
                    use_cache=False,
                    return_sub_html=True,
                )
                if top_r:
                    races = top_r
                    sub_html = raw_sub or None
                    list_meta["race_list_source"] = "race.netkeiba.com/top"
            except Exception as e:
                logger.warning("レース一覧(netkeiba top フォールバック) 失敗 [%s]: %s", date, e)

        if sub_html:
            try:
                self.archive.save("race_lists", date, sub_html)
            except Exception as e:
                logger.debug("race_lists アーカイブ(top) スキップ: %s", e)

        if not races:
            payload = {"date": date, "races": [], "_meta": {}}
            if list_meta:
                payload["_meta"].update(list_meta)
            self.storage.save("race_lists", date, payload)
            logger.info("レース一覧取得: %s - 0 レース", date)
            return []

        return _save_and_return(races, list_meta.get("race_list_source", ""))

    # ── バッチ系 ──────────────────────────────────────

    def scrape_date_results(self, date: str) -> list[dict]:
        races = self.scrape_race_list(date)
        results = []
        for race in races:
            data = self.scrape_race_result(race["race_id"])
            if data:
                results.append(data)
        logger.info("日付 %s: %d/%d レース取得完了", date, len(results), len(races))
        return results

    def scrape_date_cards(self, date: str) -> list[dict]:
        races = self.scrape_race_list(date)
        cards = []
        for race in races:
            data = self.scrape_race_card(race["race_id"])
            if data:
                cards.append(data)
        logger.info("日付 %s: %d/%d 出馬表取得完了", date, len(cards), len(races))
        return cards

    def scrape_horses_from_results(self, race_ids: list[str] | None = None) -> int:
        if race_ids is None:
            race_ids = self.storage.list_keys("race_result")

        horse_ids: set[str] = set()
        for rid in race_ids:
            data = self.storage.load("race_result", rid)
            if data:
                for entry in data.get("entries", []):
                    hid = entry.get("horse_id", "")
                    if hid:
                        horse_ids.add(hid)

        logger.info("馬情報取得開始: %d 頭", len(horse_ids))
        count = 0
        skipped = 0
        for hid in sorted(horse_ids):
            if self.is_horse_fresh(hid):
                skipped += 1
                count += 1
                continue
            if self.scrape_horse(hid):
                count += 1
        logger.info("馬情報取得完了: %d/%d 頭 (%d頭スキップ/鮮度OK)",
                    count, len(horse_ids), skipped)
        return count

    def batch_scrape_results(self, start_date: str, end_date: str):
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        total_races = 0
        current = start
        while current <= end:
            if current.weekday() in (5, 6):
                date_str = current.strftime("%Y%m%d")
                results = self.scrape_date_results(date_str)
                total_races += len(results)
            current += timedelta(days=1)
        logger.info("期間一括取得完了: %s〜%s, 合計 %d レース", start_date, end_date, total_races)

    # ── 日付指定: 全データソース一括取得 ──────────────────────

    def scrape_date_all(self, date: str, smart_skip: bool = True) -> dict:
        """日付を指定して、全レースの全データソースを一括取得する。

        smart_skip=True の場合、最新データが揃っているレースはスキップされる。
        レース間に自然なクールダウンを挿入してbot検出を回避する。
        """
        races = self.scrape_race_list(date)
        if not races:
            logger.warning("レース一覧が空です: %s", date)
            return {"date": date, "races": [], "total": 0}

        logger.info("=== %s: %d レースの全データ取得開始 (smart_skip=%s) ===",
                     date, len(races), smart_skip)
        results = []
        total_skipped = 0
        for i, race in enumerate(races):
            rid = race["race_id"]
            logger.info("--- [%d/%d] %s %s ---", i + 1, len(races),
                        rid, race.get("race_name", ""))
            collected = self.scrape_race_all(rid, smart_skip=smart_skip)
            skipped = len(collected.get("skipped", []))
            total_skipped += skipped
            results.append(collected.get("summary", {}))

            if i < len(races) - 1:
                inter_race_pause = random.uniform(3.0, 8.0)
                logger.debug("レース間クールダウン: %.1f 秒", inter_race_pause)
                time.sleep(inter_race_pause)

        # SmartRC は scrape_race_all 内で各レースごとに取得済み

        logger.info("=== %s: 全 %d レース取得完了 (スキップ: %d カテゴリ) ===",
                     date, len(races), total_skipped)
        return {"date": date, "races": results, "total": len(results),
                "skipped_categories": total_skipped}

    def batch_scrape_all(self, start_date: str, end_date: str,
                         weekends_only: bool = True):
        """期間指定で全データソースを一括取得する。日付間に長めの休止を入れる。"""
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        total_races = 0
        dates_processed = 0
        current = start
        while current <= end:
            if not weekends_only or current.weekday() in (5, 6):
                date_str = current.strftime("%Y%m%d")
                result = self.scrape_date_all(date_str)
                total_races += result.get("total", 0)
                dates_processed += 1

                if current < end:
                    inter_date_pause = random.uniform(30.0, 90.0)
                    logger.info("日付間クールダウン: %.0f 秒 (%d 日目完了)",
                                inter_date_pause, dates_processed)
                    time.sleep(inter_date_pause)

            current += timedelta(days=1)
        logger.info("期間全データ取得完了: %s〜%s, 合計 %d レース",
                     start_date, end_date, total_races)

    # ── 効率的バッチ: 2フェーズ (レースデータ→馬一括) ─────────

    def batch_scrape_dates(
        self,
        dates: list[str] | None = None,
        race_map: dict[str, list[str]] | None = None,
        *,
        year: str = "",
        month: int = 0,
        smart_skip: bool = False,
        inter_race_sleep: tuple[float, float] = (2.0, 5.0),
        inter_date_sleep: tuple[float, float] = (20.0, 45.0),
        callback: callable | None = None,
    ) -> dict:
        """
        大規模バッチ向け効率スクレイピング。

        2フェーズ構成:
          Phase 1: 全日付×全レースのレースデータ取得 (馬情報はスキップ)
          Phase 2: 全レースから抽出した馬IDを重複排除し、一括取得

        これにより:
          - 同一馬がN回出走しても 1回の取得 + 0回のGCSチェックで済む
          - 馬タイムスタンプのプリフェッチで GCS API コールを削減
          - セッションキャッシュにより同一バッチ内は瞬時に判定

        Args:
            dates: YYYYMMDD のリスト (race_map 未指定時に使用)
            race_map: {date: [race_id, ...]} の辞書。指定時は dates は無視
            year: プリフェッチ対象年 (空欄時は全日付から推定)
            month: 月指定 (1-12)。year と組み合わせて自動生成
            smart_skip: レースデータの鮮度チェック
            inter_race_sleep: レース間待機の (min, max) 秒
            inter_date_sleep: 日付間待機の (min, max) 秒
            callback: 進捗コールバック (phase, detail)

        Returns:
            バッチ結果のサマリー辞書
        """
        def _cb(phase: str, detail: str = ""):
            if callback:
                callback(phase, detail)

        # ── 日付とレースIDの解決 ──
        if race_map is None:
            if dates is None:
                if year and month:
                    dates = self._generate_month_dates(int(year), month)
                else:
                    raise ValueError(
                        "dates, race_map, または year+month のいずれかを指定してください"
                    )
            race_map = {}
            for d in sorted(dates):
                races = self.scrape_race_list(d)
                if races:
                    race_map[d] = [r["race_id"] for r in races]
                    logger.info("レース一覧: %s → %d レース", d, len(race_map[d]))
                else:
                    logger.warning("レース一覧が空です: %s", d)

        all_dates = sorted(race_map.keys())
        total_races = sum(len(v) for v in race_map.values())

        if not all_dates:
            logger.warning("対象レースが見つかりません")
            return {"dates": 0, "races": 0, "horses_total": 0,
                    "horses_fetched": 0, "horses_skipped": 0, "errors": 0}

        # ── プリフェッチ: 馬タイムスタンプの一括取得 ──
        prefetch_years = set()
        if year:
            prefetch_years.add(str(year))
        for d in all_dates:
            prefetch_years.add(d[:4])
        for y in prefetch_years:
            self.prefetch_horse_timestamps(y)

        logger.info("=" * 60)
        logger.info("バッチスクレイピング開始 (2フェーズ)")
        logger.info("  対象: %d 開催日 / %d レース", len(all_dates), total_races)
        logger.info("  期間: %s 〜 %s", all_dates[0], all_dates[-1])
        logger.info("  馬タイムスタンプ: %d件プリフェッチ済み",
                     len(self._horse_session_cache))
        logger.info("=" * 60)

        # ── Phase 1: レースデータ取得 (馬情報は後回し) ──
        _cb("phase1", "Phase 1: レースデータ取得開始")
        logger.info(">>> Phase 1: レースデータ取得 (馬情報なし)")
        start_time = time.time()
        race_results: dict[str, dict] = {}
        horse_ids_by_race: dict[str, tuple[list[str], str]] = {}
        errors = 0
        race_idx = 0

        for di, date in enumerate(all_dates):
            race_ids = race_map[date]
            logger.info("[日付 %d/%d] %s: %d レース",
                        di + 1, len(all_dates), date, len(race_ids))

            for ri, race_id in enumerate(race_ids):
                race_idx += 1
                _cb("phase1", f"[{race_idx}/{total_races}] {race_id}")
                try:
                    collected = self._scrape_race_data_only(
                        race_id, smart_skip=smart_skip
                    )
                    race_results[race_id] = collected

                    horse_ids = []
                    race_date = collected.get("_race_date", "")
                    for src_key in ("race_card", "race_result"):
                        src = collected.get(src_key)
                        if src:
                            for entry in src.get("entries", []):
                                hid = entry.get("horse_id", "")
                                if hid and hid not in horse_ids:
                                    horse_ids.append(hid)
                    horse_ids_by_race[race_id] = (horse_ids, race_date)

                except Exception as e:
                    logger.error("レースデータ取得失敗 [%s]: %s", race_id, e)
                    errors += 1

                if ri < len(race_ids) - 1:
                    time.sleep(random.uniform(*inter_race_sleep))

            if di < len(all_dates) - 1:
                elapsed = time.time() - start_time
                done_pct = race_idx / total_races * 100
                logger.info("  Phase 1 進捗: %d/%d (%.1f%%) 経過: %.1f分",
                            race_idx, total_races, done_pct, elapsed / 60)
                time.sleep(random.uniform(*inter_date_sleep))

        phase1_time = time.time() - start_time
        logger.info("Phase 1 完了: %d レース, 経過 %.1f分",
                     len(race_results), phase1_time / 60)

        # ── Phase 2: 馬情報の重複排除→一括取得 ──
        _cb("phase2", "Phase 2: 馬情報の一括取得開始")
        logger.info(">>> Phase 2: 馬情報の一括取得")

        all_horses: dict[str, str] = {}
        for race_id, (hids, rdate) in horse_ids_by_race.items():
            for hid in hids:
                if hid not in all_horses or (rdate and rdate > all_horses[hid]):
                    all_horses[hid] = rdate

        logger.info("  ユニーク馬数: %d頭 (延べ %d頭)",
                     len(all_horses),
                     sum(len(h[0]) for h in horse_ids_by_race.values()))

        horses_fetched = 0
        horses_skipped = 0
        horse_data_map: dict[str, dict] = {}
        phase2_start = time.time()

        for i, (hid, latest_race_date) in enumerate(all_horses.items()):
            if self.is_horse_fresh(hid, latest_race_date):
                cached = self.storage.load("horse_result", hid)
                if cached:
                    horse_data_map[hid] = cached
                horses_skipped += 1
            else:
                if (i + 1) % 50 == 0 or i == 0:
                    elapsed2 = time.time() - phase2_start
                    logger.info("  馬情報 %d/%d (skip=%d, fetch=%d) 経過: %.1f分",
                                i + 1, len(all_horses),
                                horses_skipped, horses_fetched, elapsed2 / 60)
                _cb("phase2", f"馬情報 ({i+1}/{len(all_horses)}): {hid}")
                data = self.scrape_horse(hid, skip_existing=False, with_history=True)
                if data:
                    horse_data_map[hid] = data
                    horses_fetched += 1
                time.sleep(random.uniform(0.5, 1.5))

        phase2_time = time.time() - phase2_start
        logger.info("Phase 2 完了: %d頭取得, %d頭スキップ, 経過 %.1f分",
                     horses_fetched, horses_skipped, phase2_time / 60)

        # ── 結果をレース単位に結合 ──
        for race_id, collected in race_results.items():
            hids, _ = horse_ids_by_race.get(race_id, ([], ""))
            collected["horses"] = {
                hid: horse_data_map[hid]
                for hid in hids if hid in horse_data_map
            }

        total_time = time.time() - start_time
        summary = {
            "dates": len(all_dates),
            "races": len(race_results),
            "horses_total": len(all_horses),
            "horses_fetched": horses_fetched,
            "horses_skipped": horses_skipped,
            "errors": errors,
            "phase1_minutes": round(phase1_time / 60, 1),
            "phase2_minutes": round(phase2_time / 60, 1),
            "total_minutes": round(total_time / 60, 1),
            "race_results": race_results,
        }

        logger.info("=" * 60)
        logger.info("バッチ完了!")
        logger.info("  %d日 / %dレース / %d頭 (取得=%d, スキップ=%d)",
                     len(all_dates), len(race_results), len(all_horses),
                     horses_fetched, horses_skipped)
        logger.info("  所要時間: %.1f分 (Phase1: %.1f分, Phase2: %.1f分)",
                     total_time / 60, phase1_time / 60, phase2_time / 60)
        logger.info("=" * 60)

        _cb("done", f"バッチ完了: {len(race_results)}レース, {len(all_horses)}頭")
        return summary

    def _scrape_race_data_only(
        self, race_id: str, smart_skip: bool = False,
    ) -> dict:
        """
        scrape_race_all から馬情報取得を除外したレースデータのみの取得。
        HTML + API を並列実行して高速化。Phase 1 (batch_scrape_dates) で使用。
        """
        collected: dict = {"race_id": race_id, "skipped": []}

        TASK_KEY_MAP = {
            "race_shutuba": "race_card",
            "race_index": "speed_index",
            "race_shutuba_past": "shutuba_past",
            "race_odds": "odds",
            "race_pair_odds": "pair_odds",
            "race_paddock": "paddock",
            "race_barometer": "barometer",
            "race_oikiri": "oikiri",
            "race_trainer_comment": "trainer_comment",
        }

        HTML_TASKS = [
            ("race_result",           "scrape_race_result",         "レース結果"),
            ("race_shutuba",          "scrape_race_card",           "出馬表"),
            ("race_index",            "scrape_speed_index",         "タイム指数"),
            ("race_shutuba_past",     "scrape_shutuba_past",        "馬柱・調教"),
            ("race_paddock",          "scrape_paddock",             "パドック評価"),
            ("race_oikiri",           "scrape_oikiri",              "追い切り"),
            ("race_trainer_comment",  "scrape_trainer_comment",     "厩舎コメント"),
        ]
        API_TASKS = [
            ("race_odds",      "_scrape_odds_with_session"),
            ("race_pair_odds", "_scrape_pair_odds_with_session"),
            ("race_barometer", "_scrape_barometer_with_session"),
        ]

        def _run_html_group():
            results = {}
            skipped = []
            rd = ""
            for category, method_name, _label in HTML_TASKS:
                collect_key = TASK_KEY_MAP.get(category, category)
                if smart_skip and self.storage.is_fresh(category, race_id, rd):
                    results[collect_key] = self.storage.load(category, race_id)
                    skipped.append(category)
                else:
                    method = getattr(self, method_name)
                    results[collect_key] = method(race_id, skip_existing=False)
                if not rd:
                    src = results.get(collect_key)
                    if src and isinstance(src, dict):
                        raw = src.get("date", "")
                        if raw:
                            rd = str(raw).replace("-", "").replace("/", "")[:8]
            return results, skipped, rd

        def _run_api_group():
            api_session = self._create_api_session()
            results = {}
            skipped = []
            try:
                for cat, method_name in API_TASKS:
                    collect_key = TASK_KEY_MAP.get(cat, cat)
                    if smart_skip and self.storage.is_fresh(cat, race_id, ""):
                        results[collect_key] = self.storage.load(cat, race_id)
                        skipped.append(cat)
                    else:
                        method = getattr(self, method_name)
                        results[collect_key] = method(api_session, race_id)
                    time.sleep(random.uniform(0.5, 1.5))
                return results, skipped
            except Exception as e:
                logger.error("API グループ取得失敗: %s", e)
                return results, skipped
            finally:
                api_session.close()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="rdata",
        ) as pool:
            html_future = pool.submit(_run_html_group)
            api_future = pool.submit(_run_api_group)

            try:
                html_results, html_skipped, race_date = html_future.result()
            except Exception as e:
                logger.error("HTML グループ取得失敗: %s", e)
                html_results, html_skipped, race_date = {}, [], ""
            try:
                api_results, api_skipped = api_future.result()
            except Exception as e:
                logger.error("API グループ取得失敗: %s", e)
                api_results, api_skipped = {}, []

        collected.update(html_results)
        collected.update(api_results)
        collected["skipped"].extend(html_skipped + api_skipped)
        collected["_race_date"] = race_date

        # SmartRC (Phase 1 では逐次で十分)
        if not self.smartrc:
            self._init_smartrc()
        if self.smartrc:
            try:
                smartrc_data = self.scrape_smartrc(race_id, date=race_date)
                if smartrc_data and smartrc_data.get("runners"):
                    collected["smartrc"] = smartrc_data
                    collected["has_smartrc"] = True
                else:
                    collected["has_smartrc"] = False
            except Exception as e:
                logger.warning("SmartRC 取得失敗 [%s]: %s", race_id, e)
                collected["has_smartrc"] = False
        else:
            collected["has_smartrc"] = False

        return collected

    @staticmethod
    def _generate_month_dates(year: int, month: int) -> list[str]:
        """指定年月の土日リストを生成する。"""
        import calendar
        dates = []
        _, days_in_month = calendar.monthrange(year, month)
        for day in range(1, days_in_month + 1):
            dt = datetime(year, month, day)
            if dt.weekday() in (5, 6):
                dates.append(dt.strftime("%Y%m%d"))
        return dates

    # ── race_id 網羅スクレイピング (並列実行版) ────────────────

    def scrape_race_all(self, race_id: str, callback: callable | None = None,
                        smart_skip: bool = True) -> dict:
        """
        race_id を指定して予測に関連する全データを一括取得する。

        並列実行戦略:
          Phase 1: HTML スクレイピング (逐次) と API 取得 (別セッション) を並列実行
          Phase 2: SmartRC (別サーバー) と馬情報取得 (netkeiba) を並列実行
        """
        def _cb(phase: str, detail: str = ""):
            if callback:
                callback(phase, detail)

        collected: dict = {"race_id": race_id, "horses": {}, "skipped": []}

        TASK_KEY_MAP = {
            "race_shutuba": "race_card",
            "race_index": "speed_index",
            "race_shutuba_past": "shutuba_past",
            "race_odds": "odds",
            "race_pair_odds": "pair_odds",
            "race_paddock": "paddock",
            "race_barometer": "barometer",
            "race_oikiri": "oikiri",
            "race_trainer_comment": "trainer_comment",
        }

        HTML_TASKS = [
            ("race_result",           "scrape_race_result",         "レース結果"),
            ("race_shutuba",          "scrape_race_card",           "出馬表"),
            ("race_index",            "scrape_speed_index",         "タイム指数"),
            ("race_shutuba_past",     "scrape_shutuba_past",        "馬柱・調教"),
            ("race_paddock",          "scrape_paddock",             "パドック評価"),
            ("race_oikiri",           "scrape_oikiri",              "追い切り"),
            ("race_trainer_comment",  "scrape_trainer_comment",     "厩舎コメント"),
        ]
        API_TASKS = [
            ("race_odds",      "_scrape_odds_with_session",      "オッズ"),
            ("race_pair_odds", "_scrape_pair_odds_with_session", "2連系オッズ"),
            ("race_barometer", "_scrape_barometer_with_session", "偏差値"),
        ]

        # ── Phase 1: HTML (逐次) + API (別セッション) を並列実行 ──

        def _run_html_group():
            results = {}
            skipped = []
            rd = ""
            for category, method_name, label in HTML_TASKS:
                collect_key = TASK_KEY_MAP.get(category, category)
                if smart_skip and self.storage.is_fresh(category, race_id, rd):
                    _cb(collect_key, f"{label}: 最新 — スキップ")
                    results[collect_key] = self.storage.load(category, race_id)
                    skipped.append(category)
                    logger.info("スキップ (最新): %s/%s", category, race_id)
                else:
                    _cb(collect_key, f"{label}を取得中...")
                    method = getattr(self, method_name)
                    results[collect_key] = method(race_id, skip_existing=False)
                if not rd:
                    src = results.get(collect_key)
                    if src and isinstance(src, dict):
                        raw = src.get("date", "")
                        if raw:
                            rd = str(raw).replace("-", "").replace("/", "")[:8]
            return results, skipped, rd

        def _run_api_group():
            api_session = self._create_api_session()
            results = {}
            skipped = []
            try:
                for cat, method_name, label in API_TASKS:
                    collect_key = TASK_KEY_MAP.get(cat, cat)
                    if smart_skip and self.storage.is_fresh(cat, race_id, ""):
                        results[collect_key] = self.storage.load(cat, race_id)
                        skipped.append(cat)
                        logger.info("スキップ (最新): %s/%s", cat, race_id)
                    else:
                        _cb(collect_key, f"{label}を取得中...")
                        method = getattr(self, method_name)
                        results[collect_key] = method(api_session, race_id)
                    time.sleep(random.uniform(0.5, 1.5))
                return results, skipped
            except Exception as e:
                logger.error("API グループ取得失敗: %s", e)
                return results, skipped
            finally:
                api_session.close()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="phase1",
        ) as pool:
            html_future = pool.submit(_run_html_group)
            api_future = pool.submit(_run_api_group)

            try:
                html_results, html_skipped, race_date = html_future.result()
            except Exception as e:
                logger.error("HTML グループ取得失敗: %s", e)
                html_results, html_skipped, race_date = {}, [], ""
            try:
                api_results, api_skipped = api_future.result()
            except Exception as e:
                logger.error("API グループ取得失敗: %s", e)
                api_results, api_skipped = {}, []

        collected.update(html_results)
        collected.update(api_results)
        collected["skipped"].extend(html_skipped + api_skipped)

        # ── Phase 2: SmartRC + 馬情報取得を並列実行 ──

        horse_ids: list[str] = []
        for source in (collected.get("race_card"), collected.get("race_result")):
            if source:
                for entry in source.get("entries", []):
                    hid = entry.get("horse_id", "")
                    if hid and hid not in horse_ids:
                        horse_ids.append(hid)

        def _run_smartrc():
            if not self.smartrc:
                self._init_smartrc()
            if not self.smartrc:
                return None, False
            _cb("smartrc", "SmartRC データを取得中...")
            try:
                data = self.scrape_smartrc(race_id, date=race_date)
                if data and data.get("runners"):
                    n_r = len(data["runners"])
                    n_h = len(data.get("horses", {}))
                    _cb("smartrc",
                        f"SmartRC: {n_r}馬 / {n_h}血統 / 全戦績取得完了")
                    return data, True
                _cb("smartrc", "SmartRC: データなし")
                return None, False
            except Exception as e:
                logger.warning("SmartRC 取得失敗 [%s]: %s", race_id, e)
                return None, False

        def _run_horses():
            _cb("horses", f"出走馬 {len(horse_ids)}頭 の情報を取得中...")
            horses: dict[str, dict] = {}
            skipped_h: list[str] = []
            fetched = 0
            skip_count = 0
            for i, hid in enumerate(horse_ids):
                if self.is_horse_fresh(hid, race_date):
                    hdata = self.storage.load("horse_result", hid)
                    skipped_h.append(f"horse_result/{hid}")
                    skip_count += 1
                else:
                    _cb("horses", f"馬情報 ({i+1}/{len(horse_ids)}): {hid}")
                    hdata = self.scrape_horse(
                        hid, skip_existing=False, with_history=True,
                    )
                    fetched += 1
                if hdata:
                    horses[hid] = hdata
            if skip_count > 0:
                logger.info("馬情報: %d頭スキップ (鮮度OK), %d頭取得",
                            skip_count, fetched)
            return horses, skipped_h

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="phase2",
        ) as pool:
            smartrc_future = pool.submit(_run_smartrc)
            horse_future = pool.submit(_run_horses)

            try:
                smartrc_data, has_smartrc = smartrc_future.result()
            except Exception as e:
                logger.warning("SmartRC 並列取得エラー: %s", e)
                smartrc_data, has_smartrc = None, False
            try:
                horses, horse_skipped = horse_future.result()
            except Exception as e:
                logger.error("馬情報並列取得エラー: %s", e)
                horses, horse_skipped = {}, []

        collected["horses"] = horses
        collected["skipped"].extend(horse_skipped)
        if smartrc_data:
            collected["smartrc"] = smartrc_data

        # ── Phase 2b: 5世代血統 (一度取得済みならスキップ) ──
        if horse_ids:
            ped5_fetched = 0
            ped5_skipped = 0
            ped5_failed = 0
            sire_ids: set[str] = set()
            for hid in horse_ids:
                ex = self.storage.load("horse_pedigree_5gen", hid)
                if ex and len((ex.get("ancestors") or [])) >= 5:
                    ped5_skipped += 1
                    for a in ex.get("ancestors", []):
                        if a.get("generation") == 1 and a.get("position") == 0:
                            sid = a.get("horse_id", "")
                            if sid:
                                sire_ids.add(sid)
                            break
                    continue
                try:
                    _cb("pedigree_5gen",
                        f"5世代血統 ({ped5_fetched + ped5_failed + 1}/"
                        f"{len(horse_ids) - ped5_skipped}): {hid}")
                    rec = self.scrape_horse_pedigree_5gen(hid, skip_existing=False)
                    ped5_fetched += 1
                    if rec:
                        for a in rec.get("ancestors", []):
                            if a.get("generation") == 1 and a.get("position") == 0:
                                sid = a.get("horse_id", "")
                                if sid:
                                    sire_ids.add(sid)
                                break
                except Exception as e:
                    ped5_failed += 1
                    logger.debug("5gen取得失敗 [%s]: %s", hid, e)
            if ped5_fetched > 0 or ped5_failed > 0:
                logger.info(
                    "5世代血統(出走馬): %d頭スキップ, %d頭新規取得, %d頭失敗",
                    ped5_skipped, ped5_fetched, ped5_failed,
                )

            # ── Phase 2c: 種牡馬の5世代血統 ──
            sire_ids -= set(horse_ids)
            if sire_ids:
                sire_fetched = 0
                sire_skipped = 0
                sire_failed = 0
                sire_list = sorted(sire_ids)
                for sid in sire_list:
                    ex = self.storage.load("horse_pedigree_5gen", sid)
                    if ex and len((ex.get("ancestors") or [])) >= 5:
                        sire_skipped += 1
                        continue
                    try:
                        _cb("pedigree_5gen_sire",
                            f"種牡馬5世代血統 ({sire_fetched + sire_failed + 1}/"
                            f"{len(sire_list) - sire_skipped}): {sid}")
                        self.scrape_horse_pedigree_5gen(sid, skip_existing=False)
                        sire_fetched += 1
                    except Exception as e:
                        sire_failed += 1
                        logger.debug("5gen取得失敗(種牡馬) [%s]: %s", sid, e)
                if sire_fetched > 0 or sire_failed > 0:
                    logger.info(
                        "5世代血統(種牡馬): %d頭スキップ, %d頭新規取得, %d頭失敗",
                        sire_skipped, sire_fetched, sire_failed,
                    )

        # ── サマリー構築 ──

        n_card = len((collected.get("race_card") or {}).get("entries", []))
        n_speed = len((collected.get("speed_index") or {}).get("entries", []))
        n_past = len((collected.get("shutuba_past") or {}).get("entries", []))
        n_odds = len((collected.get("odds") or {}).get("entries", []))
        n_paddock = len((collected.get("paddock") or {}).get("entries", []))
        n_barometer = len((collected.get("barometer") or {}).get("entries", []))
        n_oikiri = len((collected.get("oikiri") or {}).get("entries", []))
        collected["summary"] = {
            "total_horses": len(horse_ids),
            "horses_scraped": len(collected["horses"]),
            "has_result": collected.get("race_result") is not None,
            "has_card": collected.get("race_card") is not None,
            "card_entries": n_card,
            "has_speed_index": collected.get("speed_index") is not None,
            "speed_entries": n_speed,
            "has_past": collected.get("shutuba_past") is not None,
            "past_entries": n_past,
            "has_odds": collected.get("odds") is not None,
            "odds_entries": n_odds,
            "has_paddock": collected.get("paddock") is not None,
            "paddock_entries": n_paddock,
            "has_barometer": collected.get("barometer") is not None,
            "barometer_entries": n_barometer,
            "has_oikiri": collected.get("oikiri") is not None,
            "oikiri_entries": n_oikiri,
            "skipped_count": len(collected.get("skipped", [])),
            "has_smartrc": has_smartrc,
        }

        race_name = ""
        for src in (collected.get("race_card"), collected.get("race_result")):
            if src:
                race_name = src.get("race_name", "")
                break

        _cb("done", f"完了: {race_name} — {len(collected['horses'])}/{len(horse_ids)}頭")
        return collected

    # ── SmartRC 単独取得 ─────────────────────────────

    def scrape_smartrc(self, race_id: str, date: str = "") -> dict | None:
        """
        SmartRC から1レースの全データ (runners + horses + fullresults) を
        取得して GCS に保存する。

        Args:
            race_id: netkeiba race_id (12桁)
            date: 開催日 (YYYYMMDD)。省略時は race_shutuba / race_result から推定。
        """
        if not self.smartrc:
            self._init_smartrc()
        if not self.smartrc:
            logger.warning("SmartRC 未設定")
            return None

        if not date:
            card = self.storage.load("race_shutuba", race_id)
            if card:
                date = (card.get("date", "") or "").replace("-", "")
            if not date:
                result_data = self.storage.load("race_result", race_id)
                if result_data:
                    date = (result_data.get("date", "") or "").replace("-", "")

        from scraper.smartrc_client import netkeiba_to_rcode
        if not date or len(date) != 8:
            logger.warning("SmartRC: 日付不明 race_id=%s", race_id)
            return None

        rcode = netkeiba_to_rcode(race_id, date)
        logger.info("SmartRC rcode=%s (race_id=%s, date=%s)", rcode, race_id, date)
        data = self.smartrc.scrape_race(rcode)
        if data and data.get("runners"):
            self.storage.save("smartrc_race", race_id, data)
            logger.info("SmartRC %s: %d runners, %d horses 保存",
                        race_id, len(data["runners"]), len(data.get("horses", {})))
        return data

    def _init_smartrc(self):
        """SmartRC クライアントを遅延初期化する。"""
        if self._smartrc_checked:
            return
        self._smartrc_checked = True
        try:
            from scraper.smartrc_client import SmartRCClient
            client = SmartRCClient()
            if client.available:
                self._smartrc_client = client
                logger.info("SmartRC 利用可能")
            else:
                logger.debug("SmartRC 接続不可")
        except Exception as e:
            logger.debug("SmartRC 初期化スキップ: %s", e)

    # ── 並列取得用ヘルパー ─────────────────────────────

    def _create_api_session(self) -> requests.Session:
        """API呼び出し用の別セッションを作成。メインセッションの認証情報を共有。"""
        session = requests.Session()
        session.headers.update(dict(self.client._session.headers))
        session.cookies.update(self.client._session.cookies)
        return session

    def _scrape_odds_with_session(
        self, session: requests.Session, race_id: str,
    ) -> dict | None:
        """別セッションを使用してオッズを JSON API から取得する。"""
        try:
            data = self.odds_parser.parse_from_api(session, race_id)
            if data.get("entries"):
                self.storage.save("race_odds", race_id, data)
                logger.info("保存: race_odds/%s (%d件)", race_id,
                            len(data["entries"]))
                return data
            else:
                logger.warning("オッズデータなし: %s", race_id)
                return None
        except Exception as e:
            logger.error("オッズ取得失敗 [%s]: %s", race_id, e)
            return None

    def _scrape_pair_odds_with_session(
        self, session: requests.Session, race_id: str,
    ) -> dict | None:
        """別セッションを使用して2連系オッズを JSON API から取得する。"""
        try:
            data = self.odds_parser.parse_pair_odds_from_api(session, race_id)
            n_umaren = len(data.get("umaren", []))
            n_wide = len(data.get("wide", []))
            n_umatan = len(data.get("umatan", []))
            if n_umaren or n_wide or n_umatan:
                self.storage.save("race_pair_odds", race_id, data)
                logger.info("保存: race_pair_odds/%s (馬連=%d, ワイド=%d, 馬単=%d)",
                            race_id, n_umaren, n_wide, n_umatan)
                return data
            else:
                logger.warning("2連系オッズデータなし: %s", race_id)
                return None
        except Exception as e:
            logger.error("2連系オッズ取得失敗 [%s]: %s", race_id, e)
            return None

    def _scrape_barometer_with_session(
        self, session: requests.Session, race_id: str,
    ) -> dict | None:
        """別セッションを使用してバロメーターを AJAX API から取得する。"""
        try:
            data = self.barometer_parser.parse_from_api(session, race_id)
            if data.get("entries"):
                self.storage.save("race_barometer", race_id, data)
                logger.info("保存: race_barometer/%s (%d件)", race_id,
                            len(data["entries"]))
                return data
            else:
                logger.warning("バロメーターデータなし: %s", race_id)
                return None
        except Exception as e:
            logger.error("バロメーター取得失敗 [%s]: %s", race_id, e)
            return None

    # ══════════════════════════════════════════════════
    # 再パース: アーカイブ HTML からパーサーを再実行し JSON を更新
    # ══════════════════════════════════════════════════

    PARSER_MAP: dict[str, tuple] = {}

    def _init_parser_map(self):
        if self.PARSER_MAP:
            return
        self.PARSER_MAP.update({
            "race_result":       (self.result_parser,    "race_id"),
            "race_shutuba":      (self.card_parser,      "race_id"),
            "race_index":        (self.speed_parser,     "race_id"),
            "race_shutuba_past": (self.past_parser,      "race_id"),
            "race_odds":         (self.odds_parser,      "race_id"),
            "race_paddock":      (self.paddock_parser,    "race_id"),
            "race_barometer":    (self.barometer_parser,  "race_id"),
            "race_oikiri":       (self.oikiri_parser,     "race_id"),
            "race_trainer_comment": (self.trainer_comment_parser, "race_id"),
        })

    def reparse_one(self, category: str, key: str) -> dict | None:
        """1件の HTML アーカイブからパーサーを再実行して JSON を更新する。"""
        self._init_parser_map()

        if category == "horse_result":
            return self._reparse_horse(key)

        if category not in self.PARSER_MAP:
            logger.error("再パース未対応カテゴリ: %s", category)
            return None

        html = self.archive.load(category, key)
        if not html:
            logger.warning("HTML アーカイブなし: %s/%s", category, key)
            return None

        parser, id_param = self.PARSER_MAP[category]
        try:
            data = parser.parse(html, **{id_param: key})
            self.storage.save(category, key, data)
            n = len(data.get("entries", []))
            logger.info("再パース完了: %s/%s (%d件)", category, key, n)
            return data
        except Exception as e:
            logger.error("再パース失敗 [%s/%s]: %s", category, key, e)
            return None

    def _reparse_horse(self, horse_id: str) -> dict | None:
        profile_html = self.archive.load("horse_profile", horse_id)
        result_html = self.archive.load("horse_result_html", horse_id)
        ped_html = self.archive.load("horse_ped", horse_id)

        if not profile_html:
            logger.warning("馬 HTML アーカイブなし: horse_profile/%s", horse_id)
            return None

        try:
            data = self.horse_parser.parse(
                profile_html, horse_id=horse_id,
                result_html=result_html, ped_html=ped_html,
            )
            if data.get("horse_name"):
                self.storage.save("horse_result", horse_id, data)
                logger.info("馬 再パース完了: %s - %s (sire=%s)",
                            horse_id, data["horse_name"], data.get("sire", "?"))
                return data
            return None
        except Exception as e:
            logger.error("馬 再パース失敗 [%s]: %s", horse_id, e)
            return None

    def reparse_all(self, category: str, year: str | None = None) -> int:
        """指定カテゴリの全 HTML アーカイブを再パースする。"""
        self._init_parser_map()

        if category == "horse_result":
            keys = self.archive.list_keys("horse_profile", year)
            label = "horse_profile → horse_result"
        else:
            keys = self.archive.list_keys(category, year)
            label = category

        logger.info("再パース開始: %s (%d件)", label, len(keys))
        success = 0
        for i, key in enumerate(keys):
            if (i + 1) % 100 == 0:
                logger.info("  再パース進捗: %d/%d", i + 1, len(keys))

            if category == "horse_result":
                result = self._reparse_horse(key)
            else:
                result = self.reparse_one(category, key)
            if result is not None:
                success += 1

        logger.info("再パース完了: %s — %d/%d 成功", label, success, len(keys))
        return success

    # ── ユーティリティ ──────────────────────────────────

    def close(self):
        self.client.close()

    def summary(self):
        gcs_label = "GCS+" if self.storage.gcs_enabled else "ローカル"
        print(f"\n📂 スクレイピングデータ概要 [{gcs_label}]")
        print("─" * 50)

        categories = [
            ("race_result",       "レース結果"),
            ("race_shutuba",      "出馬表"),
            ("race_index",        "タイム指数"),
            ("race_shutuba_past", "馬柱"),
            ("race_odds",         "オッズ"),
            ("race_paddock",      "パドック"),
            ("race_barometer",    "バロメーター"),
            ("race_oikiri",       "追い切り"),
            ("race_detail",       "レース詳細"),
            ("horse_result",      "馬情報"),
            ("race_lists",        "レース一覧"),
        ]

        html_categories = [
            ("race_result",       "レース結果"),
            ("race_shutuba",      "出馬表"),
            ("race_index",        "タイム指数"),
            ("race_shutuba_past", "馬柱"),
            ("race_odds",         "オッズ"),
            ("race_paddock",      "パドック"),
            ("race_barometer",    "バロメーター"),
            ("race_oikiri",       "追い切り"),
            ("horse_profile",     "馬プロフィール"),
            ("horse_result_html", "馬戦績HTML"),
            ("horse_ped",         "馬血統"),
        ]

        print("  [JSON データ]")
        for cat, label in categories:
            cnt = self.storage.count(cat)
            if cnt > 0:
                print(f"    {label:14s}: {cnt:>6,} 件")

        print()
        print("  [HTML アーカイブ (gzip)]")
        for cat, label in html_categories:
            cnt = len(self.archive.list_keys(cat))
            if cnt > 0:
                print(f"    {label:14s}: {cnt:>6,} 件")

        login_status = "✓ ログイン済" if self.client.is_logged_in else "✗ 未ログイン"
        print(f"\n  認証状態: {login_status}")
        print()


def main():
    parser = argparse.ArgumentParser(description="netkeiba.com スクレイパー")
    parser.add_argument("command", choices=[
        "race-result", "race-results-date", "race-card", "race-cards-date",
        "horse", "race-list", "batch-results", "horses-from-results",
        "speed-index", "shutuba-past", "race-detail",
        "race-all", "scrape-all-date", "batch-all",
        "batch-dates", "batch-month",
        "summary", "reparse", "reparse-one",
        "structure-check",
        "smartrc",
    ])
    parser.add_argument("args", nargs="*", help="race_id, date (YYYYMMDD), horse_id, category等")
    parser.add_argument("--interval", type=float, default=1.0, help="リクエスト間隔 (秒)")
    parser.add_argument("--no-cache", action="store_true", help="キャッシュを無効化")
    parser.add_argument("--no-login", action="store_true", help="ログインせずに実行")
    parser.add_argument("--year", default=None, help="再パース時の年指定 (例: 2025)")
    parser.add_argument("--month", type=int, default=0, help="月指定 (batch-month用, 1-12)")
    parser.add_argument("--smart-skip", action="store_true", help="鮮度チェックでスキップ")
    parsed = parser.parse_args()

    runner = ScraperRunner(
        interval=parsed.interval,
        cache=not parsed.no_cache,
        auto_login=not parsed.no_login,
    )

    try:
        cmd = parsed.command
        args = parsed.args

        if cmd == "race-result" and args:
            runner.scrape_race_result(args[0], skip_existing=False)
        elif cmd == "race-results-date" and args:
            runner.scrape_date_results(args[0])
        elif cmd == "race-card" and args:
            runner.scrape_race_card(args[0], skip_existing=False)
        elif cmd == "race-cards-date" and args:
            runner.scrape_date_cards(args[0])
        elif cmd == "horse" and args:
            runner.scrape_horse(args[0], skip_existing=False)
        elif cmd == "race-list" and args:
            races = runner.scrape_race_list(args[0])
            for r in races:
                v = r.get("venue", "")
                n = r.get("race_name", "")
                print(f"  {r['race_id']}  {v:>3} R{r.get('round', 0):>2}  {n}")
        elif cmd == "batch-results" and len(args) >= 2:
            runner.batch_scrape_results(args[0], args[1])
        elif cmd == "horses-from-results":
            runner.scrape_horses_from_results()
        elif cmd == "speed-index" and args:
            runner.scrape_speed_index(args[0], skip_existing=False)
        elif cmd == "shutuba-past" and args:
            runner.scrape_shutuba_past(args[0], skip_existing=False)
        elif cmd == "race-detail" and args:
            runner.scrape_race_detail(args[0])
        elif cmd == "race-all" and args:
            runner.scrape_race_all(args[0])
        elif cmd == "scrape-all-date" and args:
            runner.scrape_date_all(args[0])
        elif cmd == "batch-all" and len(args) >= 2:
            weekends = "--all-days" not in sys.argv
            runner.batch_scrape_all(args[0], args[1], weekends_only=weekends)
        elif cmd == "batch-dates" and args:
            result = runner.batch_scrape_dates(
                dates=args, smart_skip=parsed.smart_skip,
            )
            print(f"\n完了: {result['races']}レース, "
                  f"馬={result['horses_total']}頭 "
                  f"(取得={result['horses_fetched']}, "
                  f"スキップ={result['horses_skipped']}), "
                  f"エラー={result['errors']}, "
                  f"所要時間={result['total_minutes']}分")
        elif cmd == "batch-month":
            year = parsed.year or args[0] if args else ""
            month = parsed.month or (int(args[1]) if len(args) >= 2 else 0)
            if not year or not month:
                print("Usage: batch-month YEAR --month MONTH  or  batch-month YEAR MONTH")
                sys.exit(1)
            result = runner.batch_scrape_dates(
                year=year, month=month, smart_skip=parsed.smart_skip,
            )
            print(f"\n完了: {year}年{month}月 → {result['races']}レース, "
                  f"馬={result['horses_total']}頭 "
                  f"(取得={result['horses_fetched']}, "
                  f"スキップ={result['horses_skipped']}), "
                  f"所要時間={result['total_minutes']}分")
        elif cmd == "reparse" and args:
            runner.reparse_all(args[0], year=parsed.year)
        elif cmd == "reparse-one" and len(args) >= 2:
            runner.reparse_one(args[0], args[1])
        elif cmd == "structure-check":
            runner.close()
            runner = None
            from scraper.structure_monitor import run_daily_check
            race_id = args[0] if args else ""
            horse_id = args[1] if len(args) >= 2 else ""
            result = run_daily_check(
                sample_race_id=race_id,
                sample_horse_id=horse_id,
                auto_reparse=True,
            )
            severity = result.get("severity", "UNKNOWN")
            print(f"\n構造チェック結果: {severity}")
            if result.get("critical", 0) > 0:
                print(f"  CRITICAL: {result['critical']} カテゴリ")
                for r in result.get("reparsed_categories", []):
                    print(f"    再パース: {r['category']} → {r.get('count', r.get('error', '?'))}")
            if result.get("warning", 0) > 0:
                print(f"  WARNING: {result['warning']} カテゴリ")
            print(f"  レポート: {result.get('report_path', '')}")
        elif cmd == "smartrc" and args:
            data = runner.scrape_smartrc(args[0])
            if data:
                n = len(data.get("runners", []))
                h = len(data.get("horses", {}))
                f = sum(len(v) for v in data.get("fullresults", {}).values())
                print(f"SmartRC: {n} runners, {h} horses, {f} fullresults")
        elif cmd == "summary":
            runner.summary()
        else:
            parser.print_help()

    finally:
        if runner:
            runner.close()


if __name__ == "__main__":
    main()
