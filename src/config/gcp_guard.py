"""dev 環境から GCP（GCS / Cloud SQL / Cloud Tasks / BigQuery）へ触れないようにするガード。

方針: 開発PC（``KEIBA_ENV=dev``）は GCP・VPS への読み書きを禁止し、ローカルのモックデータで動かす。
設定ミス（``GCS_BUCKET`` の記入、ADC の存在、``KEIBA_DB_BACKEND=cloud_sql`` など）で
うっかり本番に届かないよう、クライアント生成箇所で必ずこの関数を通す。回避用のフラグは設けない。
"""

from __future__ import annotations

from src.config.deployment import keiba_env


class GcpAccessForbidden(RuntimeError):
    """dev 環境から GCP へアクセスしようとしたときに送出される。"""


def gcp_forbidden() -> bool:
    return keiba_env() == "dev"


def assert_gcp_allowed(what: str) -> None:
    if gcp_forbidden():
        raise GcpAccessForbidden(
            f"dev 環境では GCP へのアクセスは禁止です（{what}）。"
            " ローカルのモックデータを使ってください: make dev-mock"
        )
