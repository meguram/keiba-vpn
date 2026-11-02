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


def on_gcp() -> bool:
    """GCP の中（Cloud Run のサービス・Jobs）で動いているか。GCS と同じリージョンなので、GCS からの読み出しに転送料がかからない。

    GCP の外（ホストPC・VPS）から GCS を読むと、読んだ量に応じて「Download APAC」（約 18 円/GiB）が課金される。
    常駐ループ・周回処理のように読み出しが積み上がる処理は、``egress_billed()`` が True なら既定で止めること（CLAUDE.md の「費用の上限」）。
    """
    import os

    return bool(os.environ.get("K_SERVICE") or os.environ.get("CLOUD_RUN_JOB"))


def egress_billed() -> bool:
    """この環境から GCS を読むと転送料が課金されるか（GCS を使う設定で、かつ GCP の外）。"""
    import os

    return bool((os.environ.get("GCS_BUCKET") or "").strip()) and not on_gcp() and not gcp_forbidden()
