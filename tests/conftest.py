"""テスト共通設定。

開発PCの .env は ``KEIBA_ENV=dev``（GCP 遮断・ローカルモックで動作）だが、既存テストは
GCS/Cloud SQL/Cloud Tasks をモックして本番相当の経路を検証する。CI（KEIBA_ENV 未設定）と
同じ条件で走らせるため、既定では KEIBA_ENV を空にする（削除だと HybridStorage が .env から
``KEIBA_ENV=dev`` を再読込してしまう。空文字なら ``os.environ.setdefault`` / dotenv は上書きしない）。
dev の挙動を検証するテストは
``monkeypatch.setenv("KEIBA_ENV", "dev")`` で明示的に切り替える。
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_data_health_output(monkeypatch, tmp_path):
    """データヘルスの結果・台帳をテストごとの一時ディレクトリへ（リポジトリの data/ を汚さない）。"""
    monkeypatch.setenv("DATA_HEALTH_OUT_DIR", str(tmp_path / "_data_health"))


@pytest.fixture(autouse=True)
def _unset_keiba_env(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "")
    monkeypatch.delenv("APP_ENV", raising=False)
