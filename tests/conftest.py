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

# --- 公開リポジトリに置かない『当て込まれた係数・重み・集計表』（config/fitted_artifacts.txt）を使うテスト ---
# 新規 clone・CI には実ファイルが無いので、無ければ skip する（ホストなど実ファイルのある環境では実行される）。
_FIT_REQUIRES: dict[str, list[str]] = {
    "tests/pipeline/test_megu_v3a.py::test_committed_model_loads_and_env_switch": ["config/megu_v3a/par.csv", "config/megu_v3a/centers.json", "config/megu_v3a/month_offsets.csv"],
    "tests/scripts/test_reflect_megu_v3a_to_db.py": ["config/megu_v3a/par.csv", "config/megu_v3a/centers.json"],
    "tests/pipeline/test_pace_coherence.py": ["config/pace_coherence.json"],
    "tests/pipeline/test_position_forecast.py::test_gate_coeff_course_surface_mean_and_unknown": ["config/gate_bias.json"],
    "tests/pipeline/test_position_forecast.py::test_inner_gate_course_moves_inner_horses_forward_and_outer_back": ["config/gate_bias.json"],
    "tests/pipeline/test_position_forecast.py::test_outer_gate_course_reverses_and_flat_course_keeps_order_by_number": ["config/gate_bias.json"],
    "tests/pipeline/test_position_forecast.py::test_pace_counts_use_pre_gate_style": ["config/gate_bias.json"],
    "tests/pipeline/test_position_forecast.py::test_attach_adds_gate_bias_and_uses_response_course_when_not_given": ["config/gate_bias.json"],
    "tests/pipeline/test_tracking_difficulty_v2.py::test_pace_v2_inference_matches_training_design": ["config/pace_coherence_v2.json", "config/pace_coherence.json"],
    "tests/pipeline/test_tracking_difficulty_v2.py::test_attach_tracking_v2_adds_fields_when_enabled": ["config/pace_coherence_v2.json", "config/pace_coherence.json"],
    "tests/pipeline/test_tracking_difficulty_v2.py::test_capacity_from_past_uses_only_runs_before_the_race_date": ["config/pace_coherence_v2.json", "config/pace_coherence.json"],
}


def pytest_collection_modifyitems(config, items):  # noqa: D103
    import pathlib

    import pytest

    root = pathlib.Path(__file__).resolve().parents[1]
    for item in items:
        for key, files in _FIT_REQUIRES.items():
            if item.nodeid.startswith(key):
                missing = [f for f in files if not (root / f).exists()]
                if missing:
                    item.add_marker(pytest.mark.skip(reason=f"当て込まれた係数が公開リポジトリに無い（config/fitted_artifacts.txt）: {missing[0]} ほか"))
                break
