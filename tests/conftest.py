"""テスト共通設定。

開発PCの .env は ``KEIBA_ENV=dev``（GCP 遮断・ローカルモックで動作）だが、既存テストは
GCS/Cloud SQL/Cloud Tasks をモックして本番相当の経路を検証する。CI（KEIBA_ENV 未設定）と
同じ条件で走らせるため、既定では KEIBA_ENV を空にする（削除だと HybridStorage が .env から
``KEIBA_ENV=dev`` を再読込してしまう。空文字なら ``os.environ.setdefault`` / dotenv は上書きしない）。
dev の挙動を検証するテストは
``monkeypatch.setenv("KEIBA_ENV", "dev")`` で明示的に切り替える。
"""

import os

import pytest

# 手元・CI のテストでは、メモリの点検で自分を終わらせない（src/api/memory_guard.py。点検そのもののテストは引数で差し替える）
os.environ.setdefault("KEIBA_MEMORY_GUARD", "0")

# 開発PCの .env / .env.<KEIBA_ENV> を読まない（KEIBA_ENV=stg を指定したテストが .env.stg の挙動フラグ・会員設定に左右され、
# ホストでだけ落ちていた。2026-11-08）。.env を読む挙動そのものは dotenv_path を渡して検証する
os.environ["KEIBA_SKIP_DOTENV"] = "1"


# --- 本物の置き場を書くテストの見張り（2026-11-16: test_entries_sync が系統の台帳・父の対応表・めぐ分析を本物に書いていた） ---
# テストの開始時と終了時に、data/meta のファイル（git 管理のものと、git に無い新しいもの）の更新時刻・大きさを比べる。
# 変わっていたら、CI（GITHUB_ACTIONS）または KEIBA_TEST_WATCH_STRICT=1 ではテスト全体を失敗にし、ほかでは警告だけにする
# （ホストの共有の作業ツリーでは、root の cron の entries-sync が系統の台帳を書くことがあり、テストのせいでなくても変わるため）。
# 書く必要のあるテストは tmp_path・環境変数・monkeypatch で差し替える。手元の置き場（data/local など）は cron が書くので見張らず、
# 下の _isolate_local_outputs で環境変数を一時ディレクトリへ差し替える（import の時点で決まる定数の出力先、例えば smartrc_sire_lines.OUT・
# build_horse_sire_map の出力先・run.GRADED_PICKS_DIR は、環境変数では守れない。それらを書く処理はテストの側でスタブにする）。
def _watch_snapshot(root):
    import subprocess

    out, err = {}, None
    try:
        r = subprocess.run(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "data/meta"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            err = (r.stderr or "").strip()[:200] or f"rc={r.returncode}"
        files = r.stdout.split("\n")
    except Exception as exc:  # noqa: BLE001
        files, err = [], str(exc)
    for f in files:
        if not f:
            continue
        try:
            st = (root / f).stat()
            out[f] = (st.st_mtime_ns, st.st_size)
        except OSError:
            continue
    return out, err


def pytest_sessionstart(session):  # noqa: D103
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    snap, err = _watch_snapshot(root)
    session.config._keiba_watch = (root, snap, err)


def _watch_changed(config):
    w = getattr(config, "_keiba_watch", None)
    if not w:
        return None, None
    root, before, err = w
    if err or not before:
        return None, err or "data/meta のファイルの一覧が空"
    after, _ = _watch_snapshot(root)
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k)), None


def _watch_strict() -> bool:
    return bool(os.environ.get("GITHUB_ACTIONS")) or os.environ.get("KEIBA_TEST_WATCH_STRICT") == "1"


def pytest_sessionfinish(session, exitstatus):  # noqa: D103
    changed, err = _watch_changed(session.config)
    session.config._keiba_watch_result = (changed, err)
    if changed and _watch_strict() and session.exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # noqa: D103
    changed, err = getattr(config, "_keiba_watch_result", (None, None))
    if err:
        terminalreporter.write_line(f"WARNING 本物の置き場の見張りができませんでした: {err}")
    if changed:
        head = "ERROR" if _watch_strict() else "WARNING"
        terminalreporter.write_line(f"{head} 本物の data/meta がテスト中に書き換わりました（テストの出力先を tmp_path などに差し替えてください。"
                                    "ホストでは cron が書いた可能性もあります）:")
        for k in changed[:30]:
            terminalreporter.write_line(f"{head}   {k}")


@pytest.fixture(autouse=True)
def _isolate_local_outputs(monkeypatch, tmp_path):
    """めぐ分析の保存先・データで予想の置き場を、テストごとの一時ディレクトリへ（本物の data/local を書かない。指定するテストはそれが優先）。"""
    monkeypatch.setenv("KEIBA_RACE_ANALYSIS_DIR", str(tmp_path / "_race_analysis"))
    monkeypatch.setenv("GRADED_PICKS_DIR", str(tmp_path / "_graded_picks"))


@pytest.fixture(autouse=True)
def _isolate_data_health_output(monkeypatch, tmp_path):
    """データヘルスの結果・台帳をテストごとの一時ディレクトリへ（リポジトリの data/ を汚さない）。"""
    monkeypatch.setenv("DATA_HEALTH_OUT_DIR", str(tmp_path / "_data_health"))


@pytest.fixture(autouse=True)
def _anti_scraping_auth_detect_only():
    """config/anti_scraping.json の auth=enforce（未ログインの /api/* を 401）に、既存の API テストが左右されないようにする。
    enforce の挙動は tests/api/test_anti_scraping_app.py が自前の設定で、設定ファイル自体は test_tiered_access.py が検証する。"""
    from src.utils import anti_scraping as AS

    eng = AS.get_engine()
    saved = (eng._cfg, eng._cfg_loaded)
    eng._cfg = AS._merge(AS.DEFAULTS, {"mode": {"auth": "detect"}})
    eng._cfg_loaded = 1e18
    yield
    eng._cfg, eng._cfg_loaded = saved


@pytest.fixture(autouse=True)
def _legacy_bet_strategy_by_default(monkeypatch):
    """推奨馬券の既定は方策 target（KEIBA_BET_STRATEGY 未設定）。従来方式の既存テストは legacy に固定し、target の挙動は tests/*/test_target_bets*.py が自前で指定する。"""
    monkeypatch.setenv("KEIBA_BET_STRATEGY", "legacy")


@pytest.fixture(autouse=True)
def _unset_keiba_env(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "")
    monkeypatch.delenv("APP_ENV", raising=False)

# --- 公開リポジトリに置かない『当て込まれた係数・重み・集計表』（config/fitted_artifacts.txt）を使うテスト ---
# 新規 clone・CI には実ファイルが無いので、無ければ skip する（ホストなど実ファイルのある環境では実行される）。
_FIT_REQUIRES: dict[str, list[str]] = {
    "tests/pipeline/test_target_bets.py": ["config/odds_pred/odds_pred_config.json"],
    "tests/api/test_target_bets_api.py": ["config/odds_pred/odds_pred_config.json"],
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


_CLOUDRUN_GCS_ENV = ("KEIBA_GCS_SHARED_STATE", "KEIBA_GCS_FAIL_FAST", "KEIBA_GCS_TIMEOUT_SEC", "KEIBA_GCS_RETRY_DEADLINE_SEC",
                     "LOAD_CACHE_MAX_MB", "RACE_DETAIL_CACHE_MAX", "KEIBA_PRERENDER_MAX_MB")


@pytest.fixture(autouse=True)
def _restore_cloudrun_gcs_env():
    """src/api/cloudrun_entry.build_app() は既定値を os.environ に setdefault する。GCS の共有の状態・すぐ返す動きが、
    後のテストの HybridStorage に漏れないように、テストごとに元へ戻し、共有の状態も消す。"""
    import os

    saved = {k: os.environ.get(k) for k in _CLOUDRUN_GCS_ENV}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        from src.scraper import storage as _st

        _st._SHARED_GCS.clear()
    except Exception:  # noqa: BLE001
        pass
