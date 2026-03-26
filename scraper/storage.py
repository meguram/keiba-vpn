"""
GCS ストレージ

GCS を唯一のデータストア (source of truth) とする。
ローカルにはデータ本体を保持しない。

ローカルに保持するもの:
  - race_lists (local_only カテゴリ: カレンダー情報)
  - data/meta/status/ (GCS 存在ステータスのキャッシュ)
  - data/meta/disk_l2_weekly_access.json — ISO 週ごとのキー別参照回数
  - data/cache/ — GCS の L2。**すべてのカテゴリ**で週内アクセスが閾値以上のキーだけ書込
    （DISK_L2_MIN_WEEKLY_ACCESSES 等）。旧ファイルは DISK_CACHE_CLEANUP_MAX_AGE_SEC で定期削除
  - メモリ LRU（LOAD_CACHE_TTL_SEC / LOAD_CACHE_MAX_ENTRIES）で GCS 往復を抑制

GCS カテゴリ名で統一管理:
  race_shutuba, race_result, race_result_on_time, race_index, race_info,
  race_lap, race_oikiri, race_paddock, race_barometer, race_return,
  horse_result, horse_ped, horse_ped_line, horse_lap, race_lists

鮮度管理 (構造ベース):
  保存時に _meta.scraped_at タイムスタンプを自動付与。
  is_fresh() は StructureMonitor の versions.json と照合し、
  データが**現在のページ構造**でスクレイピングされたものかを判定する。
  ページ構造に破壊的変更があった場合、それ以前のデータは stale となる。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

JST = timezone(timedelta(hours=9))

logger = logging.getLogger("scraper.storage")


def _is_gcs_not_found(exc: BaseException) -> bool:
    """blob 不存在 (404)。exists+download の2往復を避けるため download のみにしたときの判定用。"""
    if type(exc).__name__ == "NotFound":
        return True
    code = getattr(exc, "code", None)
    if code == 404:
        return True
    s = str(exc).lower()
    return "404" in s and "not found" in s


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


class HybridStorage:
    """
    GCS プライマリストレージ。

    - save/load/exists はすべて GCS を対象
    - local_only カテゴリ (race_lists) のみローカルファイル
    - ステータスキャッシュ: data/meta/status/{date}.json で
      モニタリングボード向けの GCS 存在情報を保持
    """

    GCS_BASE = "chuou/data/preprocessed/netkeiba/pc"
    GCS_OTHERS = "chuou/data/others"

    CATEGORY_MAP = {
        "race_shutuba": "race",
        "race_result": "race",
        "race_result_on_time": "race",
        "race_index": "race",
        "race_info": "race",
        "race_lap": "race",
        "race_lap_on_time": "race",
        "race_oikiri": "race",
        "race_paddock": "race",
        "race_barometer": "race",
        "race_return": "race",
        "race_result_lap": "race",
        "race_shutuba_past": "race",
        "race_odds": "race",
        "race_pair_odds": "race",
        "race_detail": "race",
        "horse_result": "horse",
        "horse_ped": "horse",
        "horse_ped_line": "horse",
        "horse_lap": "race",
        "horse_name": "other",
        "race_lists": "local_only",
        "smartrc_race": "race",
        "race_predictions": "race",
        "race_trainer_comment": "race",
        "horse_training": "horse",
        "horse_pedigree_5gen": "horse",
        # クッション値・含水率（JRA PDF/ライブ集約）— GCS: chuou/data/others/jra_cushion/{年}.json
        "jra_cushion": "other",
    }

    # ローカルディスクキャッシュ除外カテゴリ (local_only 等)
    _LOCAL_CACHE_EXCLUDE: set[str] = {"race_lists"}

    _GCS_TIMEOUT = 5                  # GCS API コールのタイムアウト (秒)

    def __init__(
        self,
        base_dir: str = ".",
        bucket_name: str | None = None,
        **_kwargs,
    ):
        _load_env()
        self._base_dir = Path(base_dir)
        self._local_dir = self._base_dir / "data" / "local"
        self._local_dir.mkdir(parents=True, exist_ok=True)
        self._meta_dir = self._base_dir / "data" / "meta"
        self._meta_dir.mkdir(parents=True, exist_ok=True)

        self._bucket_name = bucket_name or os.environ.get("GCS_BUCKET", "")
        self._gcs_client = None
        self._gcs_bucket = None
        self._gcs_available: bool | None = None
        self._gcs_init_lock = threading.RLock()

        # load() 用インメモリ LRU キャッシュ: {cache_key: (ts, data_or_None)}
        self._load_cache: dict[str, tuple[float, dict | None]] = {}
        self._load_cache_lock = threading.Lock()
        self._gcs_call_count = 0

        # GCS 健全性トラッキング
        self._gcs_last_failure: float = 0.0
        self._gcs_backoff = 60  # GCS障害後にリトライまで待つ秒数
        self._gcs_retry = None  # _get_bucket() 内で遅延初期化

        # メモリ L1 / ディスク L2 TTL（.env で調整可）
        self._mem_cache_ttl = float(os.environ.get("LOAD_CACHE_TTL_SEC", "3600"))
        self._mem_cache_max = int(os.environ.get("LOAD_CACHE_MAX_ENTRIES", "8000"))
        self._disk_ttl_past = float(os.environ.get("DISK_L2_TTL_PAST_SEC", "172800"))
        self._disk_ttl_current = float(os.environ.get("DISK_L2_TTL_CURRENT_SEC", "43200"))

        self._disk_hot_l2_lock = threading.Lock()
        self._disk_hot_l2_state: dict[str, Any] = {}
        self._disk_hot_l2_dirty_count = 0
        self._disk_hot_l2_last_persist: float = _time.time()
        self._DISK_HOT_L2_PERSIST_INTERVAL = 60   # 秒
        self._DISK_HOT_L2_PERSIST_BATCH = 50      # 変更回数
        self._load_disk_hot_l2_state_initial()

    def _iso_week_key(self) -> str:
        d = datetime.now(JST).isocalendar()
        return f"{d[0]}-W{d[1]:02d}"

    def _disk_l2_min_weekly_accesses(self) -> int:
        raw = (
            os.environ.get("DISK_L2_MIN_WEEKLY_ACCESSES")
            or os.environ.get("DISK_HOT_L2_MIN_WEEKLY")
            or os.environ.get("PEDIGREE_LOCAL_CACHE_MIN_WEEKLY_ACCESSES")
            or "2"
        ).strip()
        try:
            n = int(raw)
        except ValueError:
            return 2
        return max(1, min(n, 1_000_000))

    def _load_disk_hot_l2_state_initial(self) -> None:
        path_new = self._meta_dir / "disk_l2_weekly_access.json"
        path_old = self._meta_dir / "pedigree_weekly_access.json"
        wk = self._iso_week_key()
        try:
            if path_new.exists():
                raw = json.loads(path_new.read_text(encoding="utf-8"))
                if raw.get("week") == wk:
                    self._disk_hot_l2_state = raw
                    return
                self._disk_hot_l2_state = {"week": wk, "counts": {}}
                path_new.write_text(
                    json.dumps(self._disk_hot_l2_state, ensure_ascii=False, indent=1),
                    encoding="utf-8",
                )
                return
            if path_old.exists():
                raw = json.loads(path_old.read_text(encoding="utf-8"))
                if raw.get("week") == wk:
                    self._disk_hot_l2_state = raw
                    path_new.write_text(
                        json.dumps(raw, ensure_ascii=False, indent=1),
                        encoding="utf-8",
                    )
                    try:
                        path_old.unlink()
                    except OSError:
                        pass
                    return
                self._disk_hot_l2_state = {"week": wk, "counts": {}}
                path_new.write_text(
                    json.dumps(self._disk_hot_l2_state, ensure_ascii=False, indent=1),
                    encoding="utf-8",
                )
                try:
                    path_old.unlink()
                except OSError:
                    pass
                return
        except Exception as e:
            logger.warning("disk_l2_weekly_access 読込失敗: %s", e)
        self._disk_hot_l2_state = {"week": wk, "counts": {}}

    def _persist_disk_hot_l2_state_unlocked(self) -> None:
        path = self._meta_dir / "disk_l2_weekly_access.json"
        try:
            payload = json.dumps(self._disk_hot_l2_state, ensure_ascii=False, indent=1)
            fd, tmp = tempfile.mkstemp(
                dir=self._meta_dir, prefix="disk_l2_weekly_", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(payload)
                Path(tmp).replace(path)
            except Exception:
                try:
                    Path(tmp).unlink(missing_ok=True)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.warning("disk_l2_weekly_access 保存失敗: %s", e)

    def _bump_disk_hot_l2_access(self, category: str, key: str) -> int:
        with self._disk_hot_l2_lock:
            wk = self._iso_week_key()
            if self._disk_hot_l2_state.get("week") != wk:
                self._disk_hot_l2_state = {"week": wk, "counts": {}}
                self._disk_hot_l2_dirty_count = 0
            counts: dict[str, Any] = self._disk_hot_l2_state.setdefault("counts", {})
            cat = counts.setdefault(category, {})
            cat[key] = int(cat.get(key, 0)) + 1
            n = int(cat[key])
            self._disk_hot_l2_dirty_count += 1
            now = _time.time()
            if (self._disk_hot_l2_dirty_count >= self._DISK_HOT_L2_PERSIST_BATCH
                    or (now - self._disk_hot_l2_last_persist) >= self._DISK_HOT_L2_PERSIST_INTERVAL):
                self._persist_disk_hot_l2_state_unlocked()
                self._disk_hot_l2_dirty_count = 0
                self._disk_hot_l2_last_persist = now
            return n

    def flush_weekly_access(self) -> None:
        """未永続化のアクセスカウントをディスクに書き出す。シャットダウン時に呼ぶ。"""
        with self._disk_hot_l2_lock:
            if self._disk_hot_l2_dirty_count > 0:
                self._persist_disk_hot_l2_state_unlocked()
                self._disk_hot_l2_dirty_count = 0
                self._disk_hot_l2_last_persist = _time.time()

    def _weekly_disk_l2_maybe_write(
        self, category: str, key: str, data: dict[str, Any]
    ) -> None:
        """GCS と同じキーについて週次参照回数を増やし、閾値以上ならディスク L2 にだけ書く。"""
        n = self._bump_disk_hot_l2_access(category, key)
        if n >= self._disk_l2_min_weekly_accesses():
            self._write_local_cache(category, key, data)

    @property
    def gcs_enabled(self) -> bool:
        if self._gcs_available is None:
            with self._gcs_init_lock:
                if self._gcs_available is None:
                    self._gcs_available = bool(self._bucket_name)
                    if self._gcs_available:
                        try:
                            _ = self._get_bucket()
                        except Exception as e:
                            logger.warning("GCS 接続失敗 (ローカルのみモード): %s", e)
                            self._gcs_available = False
        return self._gcs_available

    @property
    def _gcs_healthy(self) -> bool:
        """GCS が最近失敗していなければ True。バックオフ期間中は False。"""
        if self._gcs_last_failure == 0.0:
            return True
        return (_time.time() - self._gcs_last_failure) > self._gcs_backoff

    def _mark_gcs_failure(self):
        if self._gcs_last_failure == 0.0:
            logger.warning("GCS 障害検知 — %d秒間ローカルキャッシュモードへ移行",
                           self._gcs_backoff)
        self._gcs_last_failure = _time.time()

    def _mark_gcs_success(self):
        if self._gcs_last_failure > 0:
            logger.info("GCS 復旧確認")
            self._gcs_last_failure = 0.0

    def check_gcs_connectivity(self, quick: bool = True) -> bool:
        """GCS 疎通確認。ソケットレベルで storage.googleapis.com:443 の最初のIPに接続テスト。"""
        if not self.gcs_enabled:
            return False
        import socket
        timeout = 2 if quick else self._GCS_TIMEOUT
        try:
            infos = socket.getaddrinfo(
                "storage.googleapis.com", 443, socket.AF_INET,
                socket.SOCK_STREAM)
            if not infos:
                self._mark_gcs_failure()
                return False
            addr = infos[0][4]  # (ip, port)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(addr)
            sock.close()
            self._mark_gcs_success()
            return True
        except (socket.timeout, OSError):
            self._mark_gcs_failure()
            return False

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
            "auth_uri": os.environ.get("GCS_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
            "token_uri": os.environ.get("GCS_TOKEN_URI", "https://oauth2.googleapis.com/token"),
            "auth_provider_x509_cert_url": os.environ.get("GCS_AUTH_PROVIDER_CERT_URL", ""),
            "client_x509_cert_url": os.environ.get("GCS_CLIENT_CERT_URL", ""),
            "universe_domain": os.environ.get("GCS_UNIVERSE_DOMAIN", "googleapis.com"),
        }
        from google.oauth2 import service_account
        return service_account.Credentials.from_service_account_info(info)

    def _get_bucket(self):
        if self._gcs_bucket is None:
            with self._gcs_init_lock:
                if self._gcs_bucket is None:
                    from google.cloud import storage as gcs_lib
                    credentials = self._build_credentials()
                    if credentials:
                        self._gcs_client = gcs_lib.Client(
                            credentials=credentials,
                            project=credentials.project_id,
                        )
                    else:
                        self._gcs_client = gcs_lib.Client()
                    self._gcs_bucket = self._gcs_client.bucket(self._bucket_name)
                    self._gcs_retry = self._build_retry()
        return self._gcs_bucket

    @staticmethod
    def _build_retry():
        """GCS API コールのリトライ設定。デフォルト120s→10sに短縮。"""
        from google.api_core.retry import Retry
        return Retry(deadline=10, initial=0.5, maximum=2, multiplier=2)

    # ── パス構築 ─────────────────────────────────────

    def _local_path(self, category: str, key: str) -> Path:
        return self._local_dir / category / f"{key}.json"

    def _gcs_blob_path(self, category: str, key: str) -> str:
        id_type = self.CATEGORY_MAP.get(category, "race")

        if id_type == "other":
            return f"{self.GCS_OTHERS}/{category}/{key}.json"
        if id_type == "local_only":
            return ""
        if id_type == "horse":
            prefix = key[:4] if len(key) >= 4 else key
            return f"{self.GCS_BASE}/{category}/{prefix}/{key}.json"

        year = key[:4] if len(key) >= 4 else "unknown"
        return f"{self.GCS_BASE}/{category}/{year}/{key}.json"

    def _is_local_only(self, category: str) -> bool:
        return self.CATEGORY_MAP.get(category) == "local_only"

    def _local_cache_path(self, category: str, key: str) -> Path:
        """ローカルディスクキャッシュのパス。data/cache/{category}/{prefix}/{key}.json"""
        cache_dir = self._base_dir / "data" / "cache" / category
        id_type = self.CATEGORY_MAP.get(category, "race")
        if id_type == "horse":
            prefix = key[:4] if len(key) >= 4 else "_"
            return cache_dir / prefix / f"{key}.json"
        year = key[:4] if len(key) >= 4 else "_"
        return cache_dir / year / f"{key}.json"

    def _is_locally_cached(self, category: str) -> bool:
        return (category not in self._LOCAL_CACHE_EXCLUDE
                and self.CATEGORY_MAP.get(category) != "local_only")

    def _local_cache_ttl(self, key: str) -> float:
        """ディスク L2 の TTL（.env DISK_L2_TTL_*、過去年は長め）。"""
        current_year = datetime.now(JST).strftime("%Y")
        key_year = key[:4] if len(key) >= 4 else ""
        if key_year and key_year < current_year:
            return self._disk_ttl_past
        return self._disk_ttl_current

    # ── save ──────────────────────────────────────────

    def save(self, category: str, key: str, data: dict[str, Any]):
        """データを保存する。local_only カテゴリはローカル、それ以外は GCS のみ。

        自動的に _meta.scraped_at タイムスタンプを付与する。
        """
        now = _time.time()
        if "_meta" not in data:
            data["_meta"] = {}
        data["_meta"]["scraped_at"] = now
        data["_meta"]["scraped_at_jst"] = datetime.fromtimestamp(
            now, tz=JST
        ).strftime("%Y-%m-%d %H:%M:%S")

        if self._is_local_only(category):
            self._save_local(category, key, data)
            return

        gated = self._is_locally_cached(category)
        # ディスク L2 は週次アクセス閾値付きのみ（GCS 成功後に _weekly_disk_l2_maybe_write）

        if not self.gcs_enabled:
            logger.warning("GCS 未接続のため save スキップ: %s/%s", category, key)
            if gated:
                self._weekly_disk_l2_maybe_write(category, key, data)
            return

        if not self._gcs_healthy:
            logger.debug("GCS バックオフ中のため save スキップ: %s/%s", category, key)
            if gated:
                self._weekly_disk_l2_maybe_write(category, key, data)
            return

        try:
            blob_path = self._gcs_blob_path(category, key)
            blob = self._get_bucket().blob(blob_path)
            content = json.dumps(data, ensure_ascii=False, indent=1)
            blob.upload_from_string(content, content_type="application/json",
                                    timeout=self._GCS_TIMEOUT,
                                    retry=self._gcs_retry)
            logger.debug("GCS save: %s/%s", category, key)
            self._mark_gcs_success()
            self.invalidate_blob_cache(category)
            self._cache_put(f"{category}/{key}", data)
            if gated:
                self._weekly_disk_l2_maybe_write(category, key, data)
        except Exception as e:
            logger.error("GCS save 失敗 [%s/%s]: %s", category, key, e)
            self._mark_gcs_failure()
            raise

    def _save_local(self, category: str, key: str, data: dict[str, Any]):
        path = self._local_path(category, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    # ── load ──────────────────────────────────────────

    def load(self, category: str, key: str,
             bypass_cache: bool = False) -> dict[str, Any] | None:
        """データを読み込む。
        優先順位: メモリキャッシュ → ローカルディスクキャッシュ → GCS。
        local_only カテゴリはローカルファイルのみ。
        """
        if self._is_local_only(category):
            return self._load_local(category, key)

        cache_key = f"{category}/{key}"
        if not bypass_cache:
            with self._load_cache_lock:
                cached = self._load_cache.get(cache_key)
            if cached and (_time.time() - cached[0]) < self._mem_cache_ttl:
                mem = cached[1]
                if mem is not None and self._is_locally_cached(category):
                    self._bump_disk_hot_l2_access(category, key)
                return mem

        # L2: ローカルディスクキャッシュ
        if not bypass_cache and self._is_locally_cached(category):
            local_data = self._read_local_cache(category, key)
            if local_data is not None:
                self._cache_put(cache_key, local_data)
                self._bump_disk_hot_l2_access(category, key)
                return local_data

        if not self.gcs_enabled or not self._gcs_healthy:
            # GCS 障害中: stale ローカルキャッシュをフォールバック
            if self._is_locally_cached(category):
                stale = self._read_local_cache(category, key, allow_stale=True)
                if stale is not None:
                    self._cache_put(cache_key, stale)
                    self._weekly_disk_l2_maybe_write(category, key, stale)
                    return stale
            if not self.gcs_enabled:
                return None

        try:
            blob_path = self._gcs_blob_path(category, key)
            blob = self._get_bucket().blob(blob_path)
            self._gcs_call_count += 1
            try:
                content = blob.download_as_text(
                    timeout=self._GCS_TIMEOUT, retry=self._gcs_retry
                )
            except Exception as dl_e:
                if _is_gcs_not_found(dl_e):
                    self._mark_gcs_success()
                    self._cache_put(cache_key, None)
                    return None
                raise
            data = json.loads(content)
            self._mark_gcs_success()
            self._cache_put(cache_key, data)
            if self._is_locally_cached(category):
                self._weekly_disk_l2_maybe_write(category, key, data)
            return data
        except Exception as e:
            logger.warning("GCS load 失敗 [%s/%s]: %s", category, key, e)
            self._mark_gcs_failure()
            if self._is_locally_cached(category):
                stale = self._read_local_cache(category, key, allow_stale=True)
                if stale is not None:
                    logger.info("stale ローカルキャッシュで代替: %s/%s", category, key)
                    self._cache_put(cache_key, stale)
                    self._weekly_disk_l2_maybe_write(category, key, stale)
                    return stale
        return None

    def _cache_put(self, cache_key: str, data: dict | None):
        """LRU キャッシュにデータを格納する。上限超過時は最古エントリを削除。"""
        with self._load_cache_lock:
            if len(self._load_cache) >= self._mem_cache_max:
                oldest_key = min(self._load_cache, key=lambda k: self._load_cache[k][0])
                del self._load_cache[oldest_key]
            self._load_cache[cache_key] = (_time.time(), data)

    def invalidate_load_cache(self, category: str = "", key: str = ""):
        """load() キャッシュを無効化する。"""
        with self._load_cache_lock:
            if category and key:
                self._load_cache.pop(f"{category}/{key}", None)
            elif category:
                keys_to_del = [k for k in self._load_cache
                               if k.startswith(f"{category}/")]
                for k in keys_to_del:
                    del self._load_cache[k]
            else:
                self._load_cache.clear()

    def _load_local(self, category: str, key: str) -> dict[str, Any] | None:
        path = self._local_path(category, key)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, NotADirectoryError):
            return None
        except Exception:
            return None

    # ── ローカルディスクキャッシュ (GCS の L2) ────────────

    def _write_local_cache(self, category: str, key: str, data: dict):
        """GCS データのローカルコピーを書き込む。"""
        path = self._local_cache_path(category, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        except Exception as e:
            logger.debug("ローカルキャッシュ書込失敗 [%s/%s]: %s", category, key, e)

    def _read_local_cache(self, category: str, key: str,
                          allow_stale: bool = False) -> dict | None:
        """ローカルキャッシュを読む。allow_stale=True なら TTL 無視。"""
        path = self._local_cache_path(category, key)
        try:
            if not path.exists():
                return None
            if not allow_stale:
                age = _time.time() - path.stat().st_mtime
                if age > self._local_cache_ttl(key):
                    return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def invalidate_local_cache(self, category: str = "", key: str = ""):
        """ローカルディスクキャッシュを無効化する。"""
        if category and key:
            path = self._local_cache_path(category, key)
            path.unlink(missing_ok=True)
        elif category:
            cache_dir = self._base_dir / "data" / "cache" / category
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir, ignore_errors=True)
        else:
            cache_dir = self._base_dir / "data" / "cache"
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir, ignore_errors=True)

    def cleanup_disk_cache(self, *, max_age_seconds: float | None = None,
                           min_weekly_accesses: int | None = None) -> dict[str, Any]:
        """
        data/cache 配下のキャッシュファイル削除:
          1. mtime が max_age 秒より古い
          2. 週次アクセス回数が min_weekly_accesses 未満
        空サブディレクトリも削除。
        """
        root = self._base_dir / "data" / "cache"
        out: dict[str, Any] = {
            "ok": True,
            "removed_files": 0,
            "removed_bytes": 0,
            "removed_dirs": 0,
            "evicted_cold": 0,
            "max_age_seconds": None,
            "min_weekly_accesses": None,
        }
        if not root.exists():
            return out

        raw = os.environ.get("DISK_CACHE_CLEANUP_MAX_AGE_SEC", "604800")
        try:
            max_age = float(max_age_seconds if max_age_seconds is not None else raw)
        except (TypeError, ValueError):
            max_age = 604800.0
        out["max_age_seconds"] = max_age

        if min_weekly_accesses is None:
            try:
                min_weekly_accesses = int(
                    os.environ.get("DISK_L2_EVICT_MIN_WEEKLY", "10"))
            except (TypeError, ValueError):
                min_weekly_accesses = 10
        out["min_weekly_accesses"] = min_weekly_accesses

        weekly_counts = self._get_all_weekly_counts()

        now = _time.time()
        removed_f = 0
        removed_b = 0
        evicted_cold = 0

        for path in list(root.rglob("*.json")):
            if "/_blob_list/" in str(path):
                continue
            try:
                st = path.stat()
            except OSError:
                continue

            # 古いファイルの削除
            if now - st.st_mtime > max_age:
                try:
                    path.unlink()
                    removed_f += 1
                    removed_b += st.st_size
                except OSError:
                    pass
                continue

            # アクセス頻度ベースの削除
            cat, key = self._parse_cache_path(root, path)
            if cat and key:
                count = weekly_counts.get(cat, {}).get(key, 0)
                if count < min_weekly_accesses:
                    try:
                        path.unlink()
                        evicted_cold += 1
                        removed_b += st.st_size
                    except OSError:
                        pass

        removed_d = 0
        if root.exists():
            for dirpath, _dirnames, _filenames in os.walk(str(root), topdown=False):
                dp = Path(dirpath)
                try:
                    if dp.is_dir() and not any(dp.iterdir()):
                        dp.rmdir()
                        removed_d += 1
                except OSError:
                    pass

        out["removed_files"] = removed_f
        out["removed_bytes"] = removed_b
        out["removed_dirs"] = removed_d
        out["evicted_cold"] = evicted_cold
        if removed_f or evicted_cold or removed_d:
            logger.info(
                "disk cache cleanup: old=%d cold=%d bytes=%d dirs=%d "
                "(max_age=%.0fs, min_weekly=%d)",
                removed_f, evicted_cold, removed_b, removed_d,
                max_age, min_weekly_accesses,
            )
        return out

    def _get_all_weekly_counts(self) -> dict[str, dict[str, int]]:
        """現在の週次アクセスカウントを返す。"""
        with self._disk_hot_l2_lock:
            wk = self._iso_week_key()
            if self._disk_hot_l2_state.get("week") != wk:
                return {}
            return dict(self._disk_hot_l2_state.get("counts", {}))

    @staticmethod
    def _parse_cache_path(root: Path, path: Path) -> tuple[str, str]:
        """data/cache/{category}/{prefix}/{key}.json → (category, key)"""
        try:
            rel = path.relative_to(root)
            parts = rel.parts
            if len(parts) >= 3:
                return parts[0], path.stem
            if len(parts) == 2:
                return parts[0], path.stem
        except (ValueError, IndexError):
            pass
        return "", ""

    def cleanup_snapshot_files(self, max_age_seconds: float = 86400) -> int:
        """data/research/_*_snapshot_cache.jsonl.gz 等の古いスナップショットを削除。"""
        patterns = [
            self._base_dir / "data" / "research" / "_ped_snapshot_cache.jsonl.gz",
            self._base_dir / "data" / "research" / "_hr_snapshot_cache.jsonl.gz",
        ]
        removed = 0
        now = _time.time()
        for p in patterns:
            try:
                if p.exists() and (now - p.stat().st_mtime) > max_age_seconds:
                    p.unlink()
                    removed += 1
                    logger.info("snapshot cleanup: removed %s", p.name)
            except OSError:
                pass
        return removed

    # ── exists ────────────────────────────────────────

    def exists(self, category: str, key: str) -> bool:
        """データが存在するか確認する。load キャッシュ → blob リストキャッシュ → GCS。"""
        if self._is_local_only(category):
            return self._local_path(category, key).exists()

        cache_key = f"{category}/{key}"
        with self._load_cache_lock:
            cached = self._load_cache.get(cache_key)
        if cached and (_time.time() - cached[0]) < self._mem_cache_ttl:
            return cached[1] is not None

        year = key[:4] if len(key) >= 4 else "unknown"
        blob_cache_key = f"{category}/{year}"
        with self._blob_list_lock:
            blob_cached = self._blob_list_cache.get(blob_cache_key)
        if blob_cached and (_time.time() - blob_cached[0]) < self._blob_list_ttl(year):
            return key in blob_cached[1]

        return self.exists_gcs(category, key)

    def exists_gcs(self, category: str, key: str) -> bool:
        if not self.gcs_enabled or self._is_local_only(category):
            return False
        if not self._gcs_healthy:
            return False
        try:
            blob_path = self._gcs_blob_path(category, key)
            self._gcs_call_count += 1
            result = self._get_bucket().blob(blob_path).exists(
                timeout=self._GCS_TIMEOUT,
                retry=self._gcs_retry)
            self._mark_gcs_success()
            return result
        except Exception:
            self._mark_gcs_failure()
            return False

    # ── バッチ操作 (モニター高速化用) ─────────────────

    _blob_list_cache: dict[str, tuple[float, dict[str, float]]] = {}
    _blob_list_lock = threading.Lock()
    _BLOB_LIST_TTL_PAST = 3600     # 1h — 過去年 (不変)
    _BLOB_LIST_TTL_CURRENT = 60    # 1min — 当年 (モニターリアルタイム反映)

    def _blob_list_ttl(self, year: str) -> float:
        current_year = datetime.now(JST).strftime("%Y")
        if year < current_year:
            return self._BLOB_LIST_TTL_PAST
        return self._BLOB_LIST_TTL_CURRENT

    _BLOB_DISK_CACHE_TTL_PAST = 3600    # 1h — 過去年の blob リスト (ほぼ不変)
    _BLOB_DISK_CACHE_TTL_CURRENT = 60   # 1min — 当年の blob リスト

    def _blob_disk_cache_path(self, category: str, year: str) -> Path:
        return self._base_dir / "data" / "cache" / "_blob_list" / f"{category}_{year}.json"

    def _blob_disk_cache_ttl(self, year: str) -> float:
        current_year = datetime.now(JST).strftime("%Y")
        if year < current_year:
            return self._BLOB_DISK_CACHE_TTL_PAST
        return self._BLOB_DISK_CACHE_TTL_CURRENT

    def _read_blob_disk_cache(self, category: str, year: str,
                              allow_stale: bool = False) -> dict[str, float] | None:
        path = self._blob_disk_cache_path(category, year)
        try:
            if not path.exists():
                return None
            if not allow_stale:
                age = _time.time() - path.stat().st_mtime
                if age > self._blob_disk_cache_ttl(year):
                    return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_blob_disk_cache(self, category: str, year: str, data: dict[str, float]):
        path = self._blob_disk_cache_path(category, year)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"))
        except Exception:
            pass

    def batch_list_blobs(
        self, category: str, year: str,
    ) -> dict[str, float]:
        """
        カテゴリ+年で GCS blob を一括リストし、{key: updated_timestamp} を返す。
        優先順位: メモリキャッシュ → ディスクキャッシュ → GCS API。
        """
        if self._is_local_only(category):
            return {}

        cache_key = f"{category}/{year}"
        with self._blob_list_lock:
            cached = self._blob_list_cache.get(cache_key)
            if cached and (_time.time() - cached[0]) < self._blob_list_ttl(year):
                return cached[1]

        disk_cached = self._read_blob_disk_cache(category, year)
        if disk_cached is not None:
            with self._blob_list_lock:
                self._blob_list_cache[cache_key] = (_time.time(), disk_cached)
            return disk_cached

        if not self.gcs_enabled or not self._gcs_healthy:
            return {}

        try:
            id_type = self.CATEGORY_MAP.get(category, "race")
            if id_type == "other":
                prefix = f"{self.GCS_OTHERS}/{category}/"
            else:
                prefix = f"{self.GCS_BASE}/{category}/{year}/"

            result: dict[str, float] = {}
            for blob in self._get_bucket().list_blobs(
                    prefix=prefix, timeout=self._GCS_TIMEOUT,
                    retry=self._gcs_retry):
                if blob.name.endswith(".json") and blob.size and blob.size > 2:
                    stem = blob.name.rsplit("/", 1)[-1].replace(".json", "")
                    ts = blob.updated.timestamp() if blob.updated else 0.0
                    result[stem] = ts

            self._mark_gcs_success()
            with self._blob_list_lock:
                self._blob_list_cache[cache_key] = (_time.time(), result)
            self._write_blob_disk_cache(category, year, result)
            return result
        except Exception as e:
            logger.warning("batch_list_blobs 失敗 [%s/%s]: %s", category, year, e)
            self._mark_gcs_failure()
            stale = self._read_blob_disk_cache(category, year, allow_stale=True)
            if stale is not None:
                logger.info("stale blob ディスクキャッシュで代替: %s/%s", category, year)
                with self._blob_list_lock:
                    self._blob_list_cache[cache_key] = (_time.time(), stale)
                return stale
            return {}

    def invalidate_blob_cache(self, category: str = "", year: str = ""):
        """blob リストキャッシュ (メモリ + ディスク) を無効化する。"""
        with self._blob_list_lock:
            if category and year:
                self._blob_list_cache.pop(f"{category}/{year}", None)
            elif category:
                keys_to_del = [k for k in self._blob_list_cache if k.startswith(f"{category}/")]
                for k in keys_to_del:
                    del self._blob_list_cache[k]
            else:
                self._blob_list_cache.clear()

        blob_cache_dir = self._base_dir / "data" / "cache" / "_blob_list"
        if category and year:
            p = blob_cache_dir / f"{category}_{year}.json"
            p.unlink(missing_ok=True)
        elif category:
            for p in blob_cache_dir.glob(f"{category}_*.json"):
                p.unlink(missing_ok=True)
        elif blob_cache_dir.exists():
            import shutil
            shutil.rmtree(blob_cache_dir, ignore_errors=True)

    def batch_check_keys(
        self, category: str, keys: list[str],
    ) -> dict[str, float]:
        """
        指定キー群の存在 + 更新日時をバッチ取得する。
        year ごとにグループ化して batch_list_blobs を並列で呼ぶ。
        """
        if not self.gcs_enabled or self._is_local_only(category):
            return {}
        years: dict[str, list[str]] = {}
        for k in keys:
            y = k[:4] if len(k) >= 4 else "unknown"
            years.setdefault(y, []).append(k)

        from concurrent.futures import ThreadPoolExecutor
        year_blobs: dict[str, dict[str, float]] = {}
        with ThreadPoolExecutor(max_workers=min(len(years), 8)) as pool:
            futures = {
                y: pool.submit(self.batch_list_blobs, category, y)
                for y in years
            }
            for y, f in futures.items():
                year_blobs[y] = f.result()

        result: dict[str, float] = {}
        for y, ks in years.items():
            blobs = year_blobs.get(y, {})
            for k in ks:
                if k in blobs:
                    result[k] = blobs[k]
        return result

    # ── 鮮度 (freshness) — 構造ベース ─────────────────

    # データカテゴリ → StructureMonitor のプローブカテゴリのマッピング
    # None = ページ構造に依存しない (JSON API / 内部生成 / ローカル専用)
    DATA_TO_STRUCTURE_MAP: dict[str, str | None] = {
        "race_shutuba": "race_shutuba",
        "race_result": "race_result",
        "race_result_on_time": "race_result",
        "race_index": "race_index",
        "race_info": "race_result",
        "race_lap": "race_result",
        "race_lap_on_time": "race_result",
        "race_oikiri": "race_oikiri",
        "race_paddock": "race_paddock",
        "race_barometer": "race_barometer",
        "race_return": "race_result",
        "race_result_lap": "race_result",
        "race_shutuba_past": "race_shutuba_past",
        "race_odds": "race_odds",
        "race_pair_odds": None,
        "race_trainer_comment": "race_trainer_comment",
        "race_detail": "race_result",
        "horse_result": "horse_result_html",
        "horse_ped": "horse_ped",
        "horse_ped_line": "horse_ped",
        "horse_lap": "race_result",
        "horse_name": None,
        "race_lists": None,
        "smartrc_race": None,
        "race_predictions": None,
    }

    def _load_structure_versions(self) -> dict[str, dict]:
        """StructureMonitor の versions.json を読み込む。"""
        vp = self._base_dir / "data" / "meta" / "structure" / "versions.json"
        if not vp.exists():
            return {}
        try:
            return json.loads(vp.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def get_scraped_at(self, category: str, key: str) -> float:
        """データの scraped_at タイムスタンプを返す。未取得なら 0。
        load() はキャッシュ付きなので追加 GCS コールなし。"""
        data = self.load(category, key)
        if not data:
            return 0.0
        return float(data.get("_meta", {}).get("scraped_at", 0))

    def get_scraped_at_from_blob(self, category: str, key: str) -> float:
        """GCS blob の updated 時刻を返す (_meta がない旧データ向けフォールバック)。"""
        if not self.gcs_enabled or self._is_local_only(category) or not self._gcs_healthy:
            return 0.0
        try:
            blob_path = self._gcs_blob_path(category, key)
            blob = self._get_bucket().blob(blob_path)
            self._gcs_call_count += 1
            blob.reload(timeout=self._GCS_TIMEOUT,
                        retry=self._gcs_retry)
            if blob.updated:
                self._mark_gcs_success()
                return blob.updated.timestamp()
        except Exception:
            self._mark_gcs_failure()
        return 0.0

    def _resolve_scraped_at(self, category: str, key: str,
                            data: dict | None = None) -> float:
        """data が渡されていればそこから、なければ load() から scraped_at を取得。"""
        if data is None:
            data = self.load(category, key)
        if data:
            ts = float(data.get("_meta", {}).get("scraped_at", 0))
            if ts > 0:
                return ts
        return self.get_scraped_at_from_blob(category, key)

    def is_fresh(self, category: str, key: str,
                 race_date: str = "",
                 _preloaded_data: dict | None = None) -> bool:
        """
        データが現在のページ構造に準拠しているかを判定する。
        _preloaded_data を渡すと追加の GCS load() を回避できる。
        """
        if _preloaded_data is None:
            _preloaded_data = self.load(category, key)
        if _preloaded_data is None:
            return False

        structure_cat = self.DATA_TO_STRUCTURE_MAP.get(category)
        if structure_cat is None:
            return True

        scraped_at = self._resolve_scraped_at(category, key, _preloaded_data)
        if scraped_at == 0:
            return False

        versions = self._load_structure_versions()
        ver_info = versions.get(structure_cat)
        if ver_info is None:
            return True

        # version==1 means no breaking change has ever been detected;
        # all existing data is valid regardless of when it was scraped.
        if ver_info.get("version", 1) <= 1:
            return True

        changed_at = float(ver_info.get("changed_at_unix", 0))
        return scraped_at >= changed_at

    def freshness_info(self, category: str, key: str,
                       race_date: str = "") -> dict[str, Any]:
        """
        鮮度情報を詳細に返す。
        旧: exists + load + load + exists + load = 5+ GCS call
        新: load 1回 (キャッシュ付き) + ローカルファイル参照のみ
        """
        _EMPTY = {"exists": False, "fresh": False, "scraped_at": 0,
                  "scraped_at_jst": "", "age_hours": -1,
                  "structure_version": None, "structure_changed_at": ""}

        data = self.load(category, key)
        if data is None:
            return _EMPTY

        scraped_at = float(data.get("_meta", {}).get("scraped_at", 0))
        if scraped_at == 0:
            scraped_at = self.get_scraped_at_from_blob(category, key)

        fresh = self.is_fresh(category, key, race_date, _preloaded_data=data)
        now = _time.time()
        age_hours = round((now - scraped_at) / 3600, 1) if scraped_at > 0 else -1
        scraped_jst = (
            datetime.fromtimestamp(scraped_at, tz=JST).strftime("%Y-%m-%d %H:%M")
            if scraped_at > 0 else ""
        )

        structure_cat = self.DATA_TO_STRUCTURE_MAP.get(category)
        ver_info = None
        if structure_cat:
            versions = self._load_structure_versions()
            ver_info = versions.get(structure_cat)

        return {
            "exists": True,
            "fresh": fresh,
            "scraped_at": scraped_at,
            "scraped_at_jst": scraped_jst,
            "age_hours": age_hours,
            "structure_version": ver_info.get("version") if ver_info else None,
            "structure_changed_at": ver_info.get("changed_at", "") if ver_info else "",
        }

    # ── list / count ─────────────────────────────────

    def list_keys(self, category: str, year: str | None = None) -> list[str]:
        """キー一覧を返す。local_only はローカル、それ以外は GCS。"""
        if self._is_local_only(category):
            return self._list_keys_local(category)

        if self.gcs_enabled and self._gcs_healthy:
            try:
                result = self._list_keys_gcs(category, year)
                self._mark_gcs_success()
                return result
            except Exception as e:
                logger.warning("GCS list_keys 失敗: %s", e)
                self._mark_gcs_failure()
        return []

    def _list_keys_gcs(self, category: str, year: str | None = None) -> list[str]:
        prefix = f"{self.GCS_BASE}/{category}/"
        if year:
            prefix += f"{year}/"
        keys = []
        for blob in self._get_bucket().list_blobs(
                prefix=prefix, timeout=self._GCS_TIMEOUT,
                retry=self._gcs_retry):
            if blob.name.endswith(".json") and blob.size > 2:
                stem = blob.name.rsplit("/", 1)[-1].replace(".json", "")
                keys.append(stem)
        return sorted(keys)

    def _list_keys_local(self, category: str) -> list[str]:
        d = self._local_dir / category
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.json"))

    def count(self, category: str, year: str | None = None) -> int:
        return len(self.list_keys(category, year))

    def list_years(self, category: str) -> list[str]:
        if not self.gcs_enabled or not self._gcs_healthy:
            return []
        try:
            prefix = f"{self.GCS_BASE}/{category}/"
            iterator = self._get_bucket().list_blobs(
                prefix=prefix, delimiter="/", timeout=self._GCS_TIMEOUT,
                retry=self._gcs_retry)
            list(iterator)
            self._mark_gcs_success()
            years = set()
            for p in iterator.prefixes:
                y = p.rstrip("/").rsplit("/", 1)[-1]
                if y.isdigit() and len(y) == 4:
                    years.add(y)
            return sorted(years)
        except Exception:
            self._mark_gcs_failure()
            return []

    # ── レース単位のステータス (モニタリングボード用) ────────

    RACE_CATEGORIES = [
        "race_shutuba", "race_result", "race_result_on_time",
        "race_index", "race_info", "race_lap",
        "race_oikiri", "race_paddock", "race_barometer",
        "race_trainer_comment",
        "race_return", "horse_lap",
    ]

    def race_status(self, race_id: str) -> dict[str, dict[str, bool]]:
        """1レースのカテゴリ別 GCS 存在状況を返す。"""
        result = {}
        for cat in self.RACE_CATEGORIES:
            result[cat] = {
                "gcs": self.exists_gcs(cat, race_id),
            }
        return result

    def race_status_batch(self, race_ids: list[str]) -> dict[str, dict]:
        return {rid: self.race_status(rid) for rid in race_ids}

    # ── データ件数サマリ (モニタリング用) ──────────────────

    def entry_count(self, category: str, key: str) -> int:
        data = self.load(category, key)
        if not data:
            return 0
        entries = data.get("entries", [])
        if isinstance(entries, list):
            return len(entries)
        return 0

    # ── 下位互換: 旧カテゴリ名でのアクセス ──────────────

    _LEGACY_MAP = {
        "races": "race_result",
        "cards": "race_shutuba",
        "speed": "race_index",
        "past": "race_oikiri",
        "odds": "race_barometer",
        "horses": "horse_result",
        "paddock": "race_paddock",
        "barometer": "race_barometer",
        "oikiri": "race_oikiri",
        "details": "race_info",
    }

    def _resolve_category(self, category: str) -> str:
        return self._LEGACY_MAP.get(category, category)

    # ── エクスポート ─────────────────────────────────

    def export_all(self, category: str, output_path: str | None = None) -> list[dict]:
        all_data = []
        for key in self.list_keys(category):
            data = self.load(category, key)
            if data:
                all_data.append(data)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=1)
        return all_data


# ── 下位互換エイリアス ─────────────────────────────────
JsonStorage = HybridStorage
