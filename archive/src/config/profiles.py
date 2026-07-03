"""KEIBA_PROFILE 環境変数によるメモリプロファイル機構。

app.py の .env 読み込み直後に apply_profile() を呼び出すと、
既存の環境変数を上書きせずにデフォルト値を注入する。

Usage (.env):
    KEIBA_PROFILE=vps

Profiles:
    vps    - 2GB VPS 向け省メモリ設定。bloodline 先読み無効、各キャッシュ縮小。
    dev    - 開発用デフォルト（= プロファイルなしと同一）。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# プロファイル別デフォルト値（既存の env var は上書きしない）
_PROFILE_DEFAULTS: dict[str, dict[str, str]] = {
    "vps": {
        # ── HybridStorage LRU ───────────────────────────────────────
        "LOAD_CACHE_MAX_ENTRIES": "1500",    # 8000 → 1500  (−25MB)
        "LOAD_CACHE_TTL_SEC": "1800",         # 3600 → 1800  (自然蒸発加速)
        "DISK_L2_MIN_WEEKLY_ACCESSES": "3",  # ディスク L2 への書き込み閾値を上げて削減
        # ── race_detail キャッシュ ──────────────────────────────────
        "RACE_DETAIL_CACHE_MAX": "200",       # unbounded → 200 エントリ上限
        # ── bloodline 先読み (267MB) ────────────────────────────────
        "KEIBA_BLOODLINE_PRELOAD": "0",       # 起動時先読みを無効化
        # ── cron precompute は軽量なので維持 ────────────────────────
        "KEIBA_EVE_PRECOMPUTE_TRACKING": "1",
        "KEIBA_EVE_PRECOMPUTE_FINAL_ODDS": "1",
    },
}


def apply_profile() -> str | None:
    """KEIBA_PROFILE に対応するデフォルト環境変数を設定する。

    既に環境変数が設定されている場合は上書きしない（.env の明示設定が優先）。
    Returns: 適用したプロファイル名、または None（プロファイル未設定）。
    """
    profile = os.environ.get("KEIBA_PROFILE", "").strip().lower()
    if not profile or profile not in _PROFILE_DEFAULTS:
        return None

    defaults = _PROFILE_DEFAULTS[profile]
    applied: list[str] = []
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            applied.append(f"{key}={value}")

    if applied:
        logger.info(
            "KEIBA_PROFILE=%s: %d 件の環境変数をデフォルト設定 [%s]",
            profile,
            len(applied),
            ", ".join(applied),
        )
    else:
        logger.info("KEIBA_PROFILE=%s: 全設定が .env で上書き済み", profile)

    return profile


def effective_int(env_key: str, fallback: int) -> int:
    """apply_profile() 後に環境変数から int 値を取得するヘルパー。"""
    return int(os.environ.get(env_key, str(fallback)))


def effective_bool(env_key: str, fallback: bool) -> bool:
    """apply_profile() 後に環境変数から bool 値を取得するヘルパー。"""
    val = os.environ.get(env_key, "1" if fallback else "0").strip().lower()
    return val not in ("0", "false", "no", "off")
