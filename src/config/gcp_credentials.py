"""
GCPサービスアカウント認証ファイルの既定パス解決。

``GOOGLE_APPLICATION_CREDENTIALS`` が未設定でも、リポジトリ既定パスにファイルが
置かれていれば自動的に ``GOOGLE_APPLICATION_CREDENTIALS`` を設定する。これにより
google-cloud-* の各クライアント（Storage/Logging/Tasks/SQL Connector等）は
Application Default Credentials (ADC) の仕組みでそのまま認証ファイルを読む。

dev/stg/prod で別々のGCPプロジェクト・サービスアカウントを使う場合に備え、
``KEIBA_ENV``/``APP_ENV``（`src.config.deployment.keiba_env()`で正規化）に応じて
環境別ファイル ``config/gcp-service-account.<env>.json`` を優先的に探す
（例: ``config/gcp-service-account.dev.json``・``.stg.json``・``.prod.json``）。
環境別ファイルが無ければ、環境を問わない既定ファイル ``config/gcp-service-account.json``
にフォールバックする（単一GCPプロジェクトで全環境を共有する場合はこれだけでよい）。

VPS側（サービング）・GCP側（スクレイピング/ML/スケジュール実行）どちらでも、
この関数を起動時に一度呼ぶだけでGCPクライアントが疎通する前提で各モジュールを実装する。
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_GCP_CREDENTIALS_PATH = Path("config/gcp-service-account.json")


def _env_specific_credentials_path(base_dir: str | Path) -> Path:
    from src.config.deployment import keiba_env

    env = keiba_env()
    return Path(base_dir) / "config" / f"gcp-service-account.{env}.json"


def candidate_gcp_credentials_paths(base_dir: str | Path = ".") -> tuple[Path, ...]:
    """探索順（環境別 → 共通既定）でパス候補を返す。"""
    return (
        _env_specific_credentials_path(base_dir),
        Path(base_dir) / DEFAULT_GCP_CREDENTIALS_PATH,
    )


def ensure_google_application_credentials(base_dir: str | Path = ".") -> bool:
    """``GOOGLE_APPLICATION_CREDENTIALS`` 未設定時に既定パス（環境別優先）を探して設定する。

    Returns:
        認証ファイルが（既存設定または既定パスで）利用可能になったかどうか。
    """
    existing = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if existing:
        return Path(existing).is_file()

    for candidate in candidate_gcp_credentials_paths(base_dir):
        if candidate.is_file():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(candidate.resolve())
            return True
    return False


def gcp_credentials_available() -> bool:
    """現在のプロセスでGCP認証ファイルが利用可能かどうか（副作用なし）。"""
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    return bool(path) and Path(path).is_file()
