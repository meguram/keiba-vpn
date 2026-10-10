"""
実行環境（KEIBA_ENV / APP_ENV）。

- ``stg`` … ステージング。UI バッジ・``X-Keiba-Env``・``/api/health`` で識別可能。
- GCS バケットや prefix は .env で本番と同一でもよい（データ直結の検証用）。

未設定時は ``prod`` 扱い（既存デプロイの後方互換）。
"""

from __future__ import annotations

import os
from typing import Any


def keiba_env_raw() -> str:
    return (os.environ.get("KEIBA_ENV") or os.environ.get("APP_ENV") or "").strip().lower()


def keiba_env() -> str:
    """正規化した環境名: ``stg`` | ``dev`` | ``prod``。"""
    r = keiba_env_raw()
    if r in ("staging", "stg"):
        return "stg"
    if r in ("dev", "development", "local"):
        return "dev"
    if r in ("prod", "production", ""):
        return "prod"
    return r or "prod"


def require_explicit_env() -> None:
    """KEIBA_ENV が未指定（空文字含む）なら起動を拒否する。暗黙の prod 起動を防ぐ。"""
    if keiba_env_raw():
        return
    if os.environ.get("KEIBA_ALLOW_IMPLICIT_PROD", "").strip() == "1":
        return
    raise SystemExit(
        "KEIBA_ENV が未設定（または空文字）のため起動を中止しました（未設定は prod 扱いになるため）。\n"
        "  .env か環境変数で KEIBA_ENV=dev|stg|prod を明示してください。\n"
        "  空文字の KEIBA_ENV が export されていると .env の値も上書きされるので unset してください。\n"
        "  意図的に暗黙 prod で起動する場合のみ KEIBA_ALLOW_IMPLICIT_PROD=1。"
    )


def is_staging() -> bool:
    return keiba_env() == "stg"


def is_production() -> bool:
    return keiba_env() == "prod"


def deployment_banner_label() -> str:
    """ナビ等に出す短いラベル。stg 以外は空文字。"""
    if not is_staging():
        return ""
    return (os.environ.get("KEIBA_DEPLOYMENT_LABEL") or "STG").strip().upper()


def keiba_staging_badge() -> str:
    """Jinja 用エイリアス。"""
    return deployment_banner_label()


def deployment_info() -> dict[str, Any]:
    """ヘルス・auth 状態 JSON 用（秘密情報は含めない）。"""
    return {
        "keiba_env": keiba_env(),
        "is_staging": is_staging(),
        "is_production": is_production(),
        "deployment_label": (os.environ.get("KEIBA_DEPLOYMENT_LABEL") or "").strip() or keiba_env(),
        "gcs_bucket": os.environ.get("GCS_BUCKET", ""),
        "gcs_prefix": os.environ.get("GCS_PREFIX", ""),
    }
