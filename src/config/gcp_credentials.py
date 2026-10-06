"""
GCPサービスアカウント認証情報を ``.env`` から構築する（ファイル配置は使わない）。

このプロジェクトのGCP認証は .env 管理に統一する:
  - 共通設定: ``.env``（dev）
  - 環境別オーバーレイ: ``.env.stg`` / ``.env.prod``
    （``KEIBA_ENV=stg|prod`` のとき ``src.utils.project_env.load_project_dotenv()`` が
    ``.env`` を読んだ後に上書きマージする。dev/stg/prodで別々のGCPプロジェクト・
    サービスアカウントを使う場合は、各 ``.env.<env>`` に異なる値を書けばよい）

サービスアカウントの各フィールドは、既存の ``src.scraper.storage.HybridStorage`` が
GCS接続に使っている ``GCS_*`` 環境変数（``GCS_TYPE`` / ``GCS_PROJECT_ID`` /
``GCS_PRIVATE_KEY_ID`` / ``GCS_PRIVATE_KEY`` / ``GCS_CLIENT_EMAIL`` / ``GCS_CLIENT_ID`` 等）
をそのまま再利用する。GCS・Cloud Tasks・Cloud SQL・BigQuery・Cloud Logging等、
このプロジェクトが使うGCPサービスは同一のサービスアカウントを前提にしているため、
認証情報を二重管理しない。

``config/gcp-service-account*.json`` のようなファイル配置は使わない
（.gitignoreのパターンは誤配置時の保険として残すが、積極的な利用は想定しない）。
"""

from __future__ import annotations

import os

try:
    from google.oauth2 import service_account as _service_account
except ImportError:  # pragma: no cover - google-auth未インストール環境向けの保険
    _service_account = None


def gcp_service_account_info() -> dict[str, str] | None:
    """``GCS_*`` 環境変数からサービスアカウント情報dictを組み立てる。

    ``GCS_PRIVATE_KEY`` が未設定の場合は ``None`` を返す（ADCへフォールバックする想定）。
    """
    private_key = os.environ.get("GCS_PRIVATE_KEY", "").strip()
    if not private_key:
        return None
    private_key = private_key.replace("\\n", "\n")

    return {
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


def build_gcp_credentials():
    """``.env``のサービスアカウント情報から ``google.auth.credentials.Credentials`` を構築する。

    ``GCS_PRIVATE_KEY`` 未設定時は ``None`` を返す。呼び出し側はこの場合、各クライアントの
    既定コンストラクタ（``credentials=None``）に渡してADC（Cloud Run実行時に自動付与される
    サービスアカウント等）へフォールバックすること。
    """
    from src.config.gcp_guard import assert_gcp_allowed

    assert_gcp_allowed("GCP 認証情報の構築")
    info = gcp_service_account_info()
    if not info or _service_account is None:
        return None
    return _service_account.Credentials.from_service_account_info(info)


def gcp_project_id() -> str:
    """``GCS_PROJECT_ID``（未設定時は``GCP_PROJECT_ID``）を返す。"""
    return os.environ.get("GCS_PROJECT_ID", "").strip() or os.environ.get("GCP_PROJECT_ID", "").strip()


def gcp_credentials_available() -> bool:
    """``.env``からGCPサービスアカウント認証情報を構築できるかどうか（副作用なし）。"""
    return gcp_service_account_info() is not None
