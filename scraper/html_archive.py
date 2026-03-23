"""
HTML アーカイブ

スクレイピングで取得した HTML からパーサーが使用する部分のみ抽出し、
gzip 圧縮して GCS に保持する。ページ構造の変更に対応できるよう、
CSS セレクタで指定された関連テーブル・要素を保全する。

軽量化フロー:
  生 HTML → 不要要素除去 (script/style/ad) → 必要テーブル抽出 → gzip

ストレージ最適化:
  HTML アーカイブは構造変更検出用のサンプルとしてのみ使用。
  カテゴリごとに MAX_SAMPLES_PER_CATEGORY 件のみ GCS に保持し、
  不要な容量消費を防止する。構造化 JSON (HybridStorage) は全件保持。

GCS パス:
  {bucket}/chuou/data/raw/html/{category}/{year_or_prefix}/{id}.html.gz

ローカルにはデータを保持しない (GCS のみ)。
"""

from __future__ import annotations

import gzip
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("scraper.html_archive")

GCS_RAW_BASE = "chuou/data/raw/html"

CATEGORY_ID_TYPE = {
    "race_result": "race",
    "race_shutuba": "race",
    "race_index": "race",
    "race_shutuba_past": "race",
    "race_odds": "race",
    "race_paddock": "race",
    "race_barometer": "race",
    "race_oikiri": "race",
    "race_trainer_comment": "race",
    "race_info": "race",
    "race_lap": "race",
    "race_return": "race",
    "horse_profile": "horse",
    "horse_result_html": "horse",
    "horse_ped": "horse",
}


ESSENTIAL_SELECTORS: dict[str, list[str]] = {
    "race_result": [
        "table.race_table_01", "table[class*='race_table']",
        "table[summary*='レース結果']", "#contents_liquid table",
        "div.data_intro", "dl.racedata",
        "div.RaceData01", "p[class*='RaceData']",
        "div.race_head_inner",
        "table.pay_table_01", "table[class*='pay_table']",
        "table.lap_table_01",
        "span.race_num",
    ],
    "race_shutuba": [
        "table.Shutuba_Table", "table.ShutubaTable", "table.RaceTable01",
        "table[class*='shutuba']", "table[class*='Shutuba']", "#page table",
        "div.RaceData01", "div.RaceData02", "div.RaceName",
        "span.RaceName", "h1[class*='RaceName']",
        "div[class*='RaceHeader']",
    ],
    "race_index": [
        "table[class*='SpeedIndex']", "table.nk_tb_common",
        "table.race_table_01", "table[id*='speed']",
        "#page table",
        "div.data_intro", "dl.racedata",
    ],
    "race_shutuba_past": [
        "table.Past_HorseList", "table[class*='past']",
        "table.nk_tb_common", "#page table",
        "table[id*='training']", "table.TrainingData",
        "div.data_intro", "dl.racedata",
    ],
    "race_odds": [
        "table[class*='Odds']", "table.nk_tb_common",
        "table.race_table_01", "#page table",
        "div.OddsHeader",
    ],
    "race_paddock": [
        "table[class*='paddock']", "table.nk_tb_common",
        "table.race_table_01", "#page table",
        "tr.HorseList", "div.Paddock_HorseList",
    ],
    "race_barometer": [
        "table[class*='Barometer']", "table.nk_tb_common",
        "table.race_table_01", "#page table",
    ],
    "race_oikiri": [
        "table[class*='Oikiri']", "table.nk_tb_common",
        "table.race_table_01", "#page table",
    ],
    "race_trainer_comment": [
        "table[class*='Comment']", "table.nk_tb_common",
        "table.race_table_01", "#page table",
        "div.CommentBlock", "tr.HorseList",
    ],
    "horse_profile": [
        "div.horse_title", "div.db_main_race",
        "div.db_prof_area_02", "table.blood_table",
        "table.db_prof_table", "table[class*='prof']",
        "table.db_h_race_results", "table[class*='race_table']",
    ],
    "horse_result_html": [
        "div.horse_title", "div.db_main_race",
        "div.db_prof_area_02", "table.blood_table",
        "table.db_prof_table", "table[class*='prof']",
        "table.db_h_race_results", "table[class*='race_table']",
    ],
    "horse_ped": [
        "table.blood_table", "table[class*='blood']",
        "table[summary*='血統']",
    ],
    "race_lists": [
        "a[href*='/race/']",
    ],
}

_JUNK_TAGS = {"script", "style", "noscript", "iframe"}
_JUNK_CLASS_PATTERNS = re.compile(
    r"(?:^|\s)(?:ad_|Ad_)|banner|sponsor|google"
    r"|Header_Area|Header_Inner\b|Header_Nav"
    r"|NkFooter|Sponsor|CM_|ContentNavi|SNS_|Side_",
)
_JUNK_ID_PATTERNS = re.compile(
    r"^(footer|sidebar|ad_|google_ad|sponsor)",
    re.I,
)


def trim_html(html: str, category: str) -> str:
    """
    HTML から不要部分を除去して軽量化する。

    DOM の親子関係を壊さないよう、「不要要素を除去する」アプローチを取る。
    """
    from bs4 import BeautifulSoup, Comment

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(_JUNK_TAGS):
        tag.decompose()

    body = soup.find("body")
    if body:
        for link_tag in body.find_all("link"):
            link_tag.unwrap()

    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    for el in list(soup.find_all(True)):
        if el.decomposed if hasattr(el, "decomposed") else (el.parent is None):
            continue
        cls_list = el.get("class") if el.attrs else None
        cls = " ".join(cls_list) if cls_list else ""
        eid = el.get("id", "") if el.attrs else ""
        if cls and _JUNK_CLASS_PATTERNS.search(cls):
            el.decompose()
        elif eid and _JUNK_ID_PATTERNS.search(eid):
            el.decompose()

    removable_attrs = {"onclick", "onload", "onmouseover", "onmouseout",
                       "data-src", "data-original"}
    for tag in soup.find_all(True):
        if not tag.attrs:
            continue
        for attr in list(tag.attrs.keys()):
            if attr in removable_attrs or attr.startswith("data-ad"):
                del tag[attr]

    result = str(soup)
    result = re.sub(r"\n\s*\n+", "\n", result)
    return result


def _load_env():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


class HtmlArchive:
    """
    gzip 圧縮 HTML の GCS ストレージ。

    save → gzip 圧縮 → GCS
    load → GCS から取得 → 解凍して返す
    ローカルにはデータを保持しない。

    カテゴリごとに MAX_SAMPLES_PER_CATEGORY 件のみ保持。
    構造変更検出 (structure_monitor) 用のサンプルとして十分な件数を維持しつつ、
    GCS 容量を大幅に削減する。
    """

    MAX_SAMPLES_PER_CATEGORY = 10
    _GCS_TIMEOUT = 5

    def __init__(
        self,
        bucket_name: str | None = None,
        max_samples: int | None = None,
        **_kwargs,
    ):
        _load_env()
        self._bucket_name = bucket_name or os.environ.get("GCS_BUCKET", "")
        self._gcs_client = None
        self._gcs_bucket = None
        self._gcs_available: bool | None = None

        if max_samples is not None:
            self.MAX_SAMPLES_PER_CATEGORY = max_samples

        self._session_save_count: dict[str, int] = {}
        self._category_quota_cache: dict[str, bool] = {}
        self._cleanup_done = False

    # ── GCS 接続 ──────────────────────────────────────

    @property
    def gcs_enabled(self) -> bool:
        if self._gcs_available is None:
            self._gcs_available = bool(self._bucket_name)
            if self._gcs_available:
                try:
                    _ = self._get_bucket()
                except Exception as e:
                    logger.warning("GCS (HTML archive) 接続失敗: %s", e)
                    self._gcs_available = False
        return self._gcs_available

    @staticmethod
    def _build_credentials():
        private_key = os.environ.get("GCS_PRIVATE_KEY", "")
        if not private_key:
            return None
        private_key = private_key.replace("\\n", "\n")
        info = {
            "type": os.environ.get("GCS_TYPE", "service_account"),
            "project_id": os.environ.get("GCS_PROJECT_ID", ""),
            "private_key_id": os.environ.get("GCS_PRIVATE_KEY_ID", ""),
            "private_key": private_key,
            "client_email": os.environ.get("GCS_CLIENT_EMAIL", ""),
            "client_id": os.environ.get("GCS_CLIENT_ID", ""),
            "auth_uri": os.environ.get("GCS_AUTH_URI", ""),
            "token_uri": os.environ.get("GCS_TOKEN_URI", ""),
            "auth_provider_x509_cert_url": os.environ.get("GCS_AUTH_PROVIDER_CERT_URL", ""),
            "client_x509_cert_url": os.environ.get("GCS_CLIENT_CERT_URL", ""),
            "universe_domain": os.environ.get("GCS_UNIVERSE_DOMAIN", "googleapis.com"),
        }
        from google.oauth2 import service_account
        return service_account.Credentials.from_service_account_info(info)

    def _get_bucket(self):
        if self._gcs_bucket is None:
            from google.cloud import storage as gcs_lib
            creds = self._build_credentials()
            if creds:
                self._gcs_client = gcs_lib.Client(credentials=creds, project=creds.project_id)
            else:
                self._gcs_client = gcs_lib.Client()
            self._gcs_bucket = self._gcs_client.bucket(self._bucket_name)
        return self._gcs_bucket

    # ── パス構築 ──────────────────────────────────────

    def _gcs_blob_path(self, category: str, key: str) -> str:
        id_type = CATEGORY_ID_TYPE.get(category, "race")
        if id_type == "horse":
            prefix = key[:4] if len(key) >= 4 else key
        else:
            prefix = key[:4] if len(key) >= 4 else "unknown"
        return f"{GCS_RAW_BASE}/{category}/{prefix}/{key}.html.gz"

    # ── save ──────────────────────────────────────────

    def _has_quota(self, category: str) -> bool:
        """このセッションでまだ保存枠が残っているかを判定する。"""
        session_count = self._session_save_count.get(category, 0)
        if session_count >= self.MAX_SAMPLES_PER_CATEGORY:
            return False

        if category in self._category_quota_cache:
            return self._category_quota_cache[category]

        try:
            existing = len(self._list_keys_gcs(category))
        except Exception:
            existing = 0
        has = existing < self.MAX_SAMPLES_PER_CATEGORY
        self._category_quota_cache[category] = has
        if not has:
            logger.debug("HTML archive 枠なし (既存 %d 件): %s", existing, category)
        return has

    def _auto_cleanup(self):
        """初回 save 時に既存データのクリーンアップを実行する。"""
        if self._cleanup_done:
            return
        self._cleanup_done = True
        try:
            result = self.cleanup(keep_per_category=self.MAX_SAMPLES_PER_CATEGORY)
            if result["total_deleted"] > 0:
                logger.info("自動クリーンアップ: %d ファイル削除 (%.1f MB 解放)",
                            result["total_deleted"],
                            result["total_freed_bytes"] / 1024 / 1024)
        except Exception as e:
            logger.warning("自動クリーンアップ失敗: %s", e)

    def save(self, category: str, key: str, html: str):
        """
        HTML を軽量化 → gzip 圧縮して GCS に保存する。

        カテゴリごとに MAX_SAMPLES_PER_CATEGORY 件まで。
        サンプル上限に達した場合は保存をスキップする。
        """
        if not self.gcs_enabled:
            return

        self._auto_cleanup()

        if not self._has_quota(category):
            return

        trimmed = trim_html(html, category)
        compressed = gzip.compress(trimmed.encode("utf-8"), compresslevel=6)
        raw_size = len(html.encode("utf-8"))
        trimmed_size = len(trimmed.encode("utf-8"))
        ratio = len(compressed) / raw_size * 100 if raw_size > 0 else 0

        try:
            blob = self._get_bucket().blob(self._gcs_blob_path(category, key))
            blob.upload_from_string(compressed, content_type="application/gzip",
                                    timeout=self._GCS_TIMEOUT)
            self._session_save_count[category] = (
                self._session_save_count.get(category, 0) + 1
            )
            logger.debug("HTML archive GCS: %s/%s (%dKB → %dKB → %dKB gz, %.0f%%削減)",
                         category, key,
                         raw_size // 1024, trimmed_size // 1024,
                         len(compressed) // 1024, 100 - ratio)
        except Exception as e:
            logger.warning("HTML archive GCS save 失敗 [%s/%s]: %s", category, key, e)

    # ── load ──────────────────────────────────────────

    def load(self, category: str, key: str) -> str | None:
        """GCS から HTML を読み出す。"""
        if not self.gcs_enabled:
            return None

        try:
            blob_path = self._gcs_blob_path(category, key)
            blob = self._get_bucket().blob(blob_path)
            if blob.exists(timeout=self._GCS_TIMEOUT):
                compressed = blob.download_as_bytes(timeout=self._GCS_TIMEOUT)
                return gzip.decompress(compressed).decode("utf-8")
        except Exception as e:
            logger.warning("HTML archive GCS load 失敗 [%s/%s]: %s", category, key, e)
        return None

    # ── exists ────────────────────────────────────────

    def exists(self, category: str, key: str) -> bool:
        return self.exists_gcs(category, key)

    def exists_gcs(self, category: str, key: str) -> bool:
        if not self.gcs_enabled:
            return False
        try:
            return self._get_bucket().blob(
                self._gcs_blob_path(category, key)).exists(timeout=self._GCS_TIMEOUT)
        except Exception:
            return False

    # ── list ──────────────────────────────────────────

    def list_keys(self, category: str, year: str | None = None) -> list[str]:
        if not self.gcs_enabled:
            return []
        try:
            return self._list_keys_gcs(category, year)
        except Exception:
            return []

    def _list_keys_gcs(self, category: str, year: str | None = None) -> list[str]:
        prefix = f"{GCS_RAW_BASE}/{category}/"
        if year:
            prefix += f"{year}/"
        keys = []
        for blob in self._get_bucket().list_blobs(
                prefix=prefix, timeout=self._GCS_TIMEOUT):
            if blob.name.endswith(".html.gz") and blob.size > 10:
                stem = blob.name.rsplit("/", 1)[-1].replace(".html.gz", "")
                keys.append(stem)
        return sorted(keys)

    # ── ユーティリティ ────────────────────────────────

    def stats(self, category: str, key: str) -> dict[str, Any]:
        """アーカイブの圧縮統計を返す (GCS から取得)。"""
        if not self.gcs_enabled:
            return {"exists": False}
        try:
            blob = self._get_bucket().blob(self._gcs_blob_path(category, key))
            if not blob.exists(timeout=self._GCS_TIMEOUT):
                return {"exists": False}
            compressed = blob.download_as_bytes(timeout=self._GCS_TIMEOUT)
            raw = gzip.decompress(compressed)
            return {
                "exists": True,
                "raw_bytes": len(raw),
                "compressed_bytes": len(compressed),
                "ratio_pct": round(len(compressed) / len(raw) * 100, 1) if raw else 0,
            }
        except Exception:
            return {"exists": False}

    # ── クリーンアップ ────────────────────────────────

    def cleanup(
        self,
        keep_per_category: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        既存 HTML アーカイブを整理し、カテゴリごとに最新 N 件のみ残す。

        keep_per_category: 残す件数 (デフォルト: MAX_SAMPLES_PER_CATEGORY)
        dry_run: True の場合は削除せずカウントのみ返す

        Returns: {"categories": {cat: {"kept": n, "deleted": n, "freed_bytes": n}}, "total_deleted": n}
        """
        if not self.gcs_enabled:
            logger.warning("GCS 未接続のためクリーンアップ不可")
            return {"categories": {}, "total_deleted": 0, "total_freed_bytes": 0}

        keep = keep_per_category or self.MAX_SAMPLES_PER_CATEGORY
        bucket = self._get_bucket()
        prefix = f"{GCS_RAW_BASE}/"

        blobs_by_cat: dict[str, list] = {}
        for blob in bucket.list_blobs(prefix=prefix, timeout=self._GCS_TIMEOUT):
            if not blob.name.endswith(".html.gz"):
                continue
            rel = blob.name[len(prefix):]
            cat = rel.split("/")[0]
            blobs_by_cat.setdefault(cat, []).append(blob)

        result: dict[str, Any] = {"categories": {}, "total_deleted": 0, "total_freed_bytes": 0}

        for cat in sorted(blobs_by_cat):
            cat_blobs = blobs_by_cat[cat]
            cat_blobs.sort(key=lambda b: b.updated or b.time_created, reverse=True)

            to_keep = cat_blobs[:keep]
            to_delete = cat_blobs[keep:]

            freed = sum(b.size or 0 for b in to_delete)
            cat_info = {
                "kept": len(to_keep),
                "deleted": len(to_delete),
                "freed_bytes": freed,
            }
            result["categories"][cat] = cat_info
            result["total_deleted"] += len(to_delete)
            result["total_freed_bytes"] += freed

            if dry_run:
                logger.info("[DRY-RUN] %s: 保持=%d, 削除=%d (%.1f MB)",
                            cat, len(to_keep), len(to_delete), freed / 1024 / 1024)
            else:
                for blob in to_delete:
                    try:
                        blob.delete()
                    except Exception as e:
                        logger.warning("削除失敗 [%s]: %s", blob.name, e)
                logger.info("%s: 保持=%d, 削除=%d (%.1f MB 解放)",
                            cat, len(to_keep), len(to_delete), freed / 1024 / 1024)

        self._category_quota_cache.clear()
        action = "検出 (dry-run)" if dry_run else "削除"
        logger.info("クリーンアップ完了: %d ファイル%s, %.1f MB 解放",
                     result["total_deleted"], action,
                     result["total_freed_bytes"] / 1024 / 1024)
        return result
