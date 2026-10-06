"""GCS 経由の学習済みモデル (`models/keiba_model.pkl`) 同期ユーティリティ。

背景:
    ライブ推論 (`src.pipeline.inference.race_prediction_service.build_race_prediction_response`)
    は MLflow Registry を経由せず、ローカル `models/keiba_model.pkl` を直接読み込む実装になって
    いる（無ければヒューリスティックにフォールバック）。この実装自体は変更せず、学習を GCP 側で
    行い VPS 側（サービング）へ学習済みモデルを配布するための「GCS 中継」だけをここに追加する。

    - 学習側 (GCP): 学習完了時に `upload_model_to_gcs()` を呼び、`models/keiba_model.pkl` を
      GCS へアップロードする。
    - サービング側 (VPS): 手動または定期実行で `sync_latest_model_from_gcs()` を呼び、GCS 側が
      新しい場合のみローカルファイルを更新する。

    `HybridStorage`（`src/scraper/storage.py`）は category/key 形式の JSON/Parquet 用設計のため、
    生バイナリ (pkl) の読み書きにはここでは `google.cloud.storage` を直接使う。認証ロジックのみ
    `HybridStorage._build_credentials()` を再利用する（`GCS_PRIVATE_KEY` 等が未設定の場合は
    Application Default Credentials にフォールバック）。

    ネットワーク障害・認証未設定・GCS 無効時でも例外を外に出さず False を返す
    （サービング側の既存動作を止めないことを最優先する）。
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from src.config.gcp_guard import gcp_forbidden

logger = logging.getLogger("pipeline.models.model_sync")

DEFAULT_LOCAL_MODEL_PATH = "models/keiba_model.pkl"
DEFAULT_GCS_BLOB_NAME = "models/keiba_model.pkl"


def _get_bucket_name() -> str:
    return os.environ.get("GCS_BUCKET", "").strip()


def _build_gcs_bucket():
    """GCS バケットオブジェクトを構築する。GCS 無効・認証不可時は None を返す（例外は出さない）。"""
    bucket_name = _get_bucket_name()
    if not bucket_name or gcp_forbidden():
        return None
    try:
        from google.cloud import storage as gcs_lib

        from src.scraper.storage import HybridStorage

        credentials = HybridStorage._build_credentials()
        if credentials:
            client = gcs_lib.Client(credentials=credentials, project=credentials.project_id)
        else:
            client = gcs_lib.Client()
        return client.bucket(bucket_name)
    except Exception as e:
        logger.warning("GCS バケット構築に失敗しました（同期をスキップ）: %s", e)
        return None


def _is_gcs_not_found(exc: BaseException) -> bool:
    if type(exc).__name__ == "NotFound":
        return True
    code = getattr(exc, "code", None)
    if code == 404:
        return True
    s = str(exc).lower()
    return "404" in s and "not found" in s


def sync_latest_model_from_gcs(
    local_path: str | Path = DEFAULT_LOCAL_MODEL_PATH,
    gcs_blob_name: str = DEFAULT_GCS_BLOB_NAME,
) -> bool:
    """GCS 上のモデルがローカルより新しければダウンロードしてローカルへ反映する。

    Args:
        local_path: ローカルの保存先 (既定: ``models/keiba_model.pkl``)。
        gcs_blob_name: GCS 上の blob 名 (既定: ``models/keiba_model.pkl``)。

    Returns:
        ダウンロードを実行した場合 True。以下の場合は False（例外は出さない）:
          - GCS 無効 (``GCS_BUCKET`` 未設定) またはバケット構築失敗
          - GCS 上に該当 blob が存在しない
          - ローカルファイルが GCS 側以上に新しい（同期不要）
          - ダウンロード中にエラーが発生した
    """
    local_path = Path(local_path)
    try:
        bucket = _build_gcs_bucket()
        if bucket is None:
            return False

        blob = bucket.blob(gcs_blob_name)
        try:
            blob.reload()
        except Exception as e:
            if _is_gcs_not_found(e):
                logger.info("GCS 上にモデルが存在しません: %s", gcs_blob_name)
            else:
                logger.warning("GCS blob メタデータ取得に失敗しました (%s): %s", gcs_blob_name, e)
            return False

        gcs_updated = getattr(blob, "updated", None)
        if local_path.is_file() and gcs_updated is not None:
            local_mtime_dt = datetime.fromtimestamp(local_path.stat().st_mtime, tz=timezone.utc)
            if gcs_updated <= local_mtime_dt:
                logger.info("ローカルモデルは既に最新のため同期不要: %s", local_path)
                return False

        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = local_path.with_name(local_path.name + ".tmp")
        blob.download_to_filename(str(tmp_path))
        tmp_path.replace(local_path)
        logger.info("GCS からモデルを同期しました: %s -> %s", gcs_blob_name, local_path)
        return True
    except Exception as e:
        logger.warning("モデル同期に失敗しました（既存モデルを維持します）: %s", e)
        return False


def upload_model_to_gcs(
    local_path: str | Path = DEFAULT_LOCAL_MODEL_PATH,
    gcs_blob_name: str = DEFAULT_GCS_BLOB_NAME,
) -> bool:
    """学習済みモデルを GCS へアップロードする（学習ジョブの最後に呼ぶ想定）。

    Args:
        local_path: アップロード元のローカルファイル (既定: ``models/keiba_model.pkl``)。
        gcs_blob_name: GCS 上の blob 名 (既定: ``models/keiba_model.pkl``)。

    Returns:
        アップロードに成功した場合 True。ローカルファイル不在・GCS 無効・エラー時は False
        （例外は出さない。呼び出し元の学習処理は失敗させない想定）。
    """
    local_path = Path(local_path)
    try:
        if not local_path.is_file():
            logger.warning("アップロード対象のローカルモデルが見つかりません: %s", local_path)
            return False

        bucket = _build_gcs_bucket()
        if bucket is None:
            return False

        blob = bucket.blob(gcs_blob_name)
        blob.upload_from_filename(str(local_path))
        logger.info("モデルを GCS へアップロードしました: %s -> %s", local_path, gcs_blob_name)
        return True
    except Exception as e:
        logger.warning("モデルの GCS アップロードに失敗しました: %s", e)
        return False
