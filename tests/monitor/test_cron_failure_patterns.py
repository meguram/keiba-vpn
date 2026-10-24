"""cron の失敗パターン辞書のテスト。代表文は 2026-10-16 時点のホストの実ログから採った。"""

from __future__ import annotations

import pytest

from src.monitor.cron_failure_patterns import PATTERNS, classify

# (pattern_id, 終了コード, 出力)
CASES = [
    ("yield_to_rescrape", 0, "[2026-10-16T02:00:01+09:00] SKIP backfill: 優先処理(取り直し)が動いているため見送り"),
    ("lock_held", 0, "2026-10-15T20:30:00+09:00 update_jockey_trainer_stats: skip (lock held)"),
    ("dev_env", 3, "KEIBA_ENV が dev に解決されたため、取得せずに終了する（GCS に保存されないため）。.env の KEIBA_ENV を確認すること"),
    ("access_restriction", 5, "アクセス制限を検知したため中断"),
    ("access_restriction", 1, "scraper.access_guard: AccessRestrictionDetected: https://db.netkeiba.com/ status=403"),
    ("smartrc_restricted", 1, "RuntimeError: SmartRC はアクセス制限の退避中（あと 3600 秒）"),
    ("schema_validation", 1, "src.scraper.schemas.SchemaValidationError: race_result/202606010101"),
    ("gcp_forbidden", 1, "src.config.gcp_guard.GcpAccessForbidden: dev では GCS に接続できない"),
    ("gcs_auth", 1, "google.auth.exceptions.DefaultCredentialsError: Your default credentials were not found"),
    ("missing_module", 1, "Traceback (most recent call last):\n  File \"x.py\", line 1\nModuleNotFoundError: No module named 'requests'"),
    ("disk_full", 1, "OSError: [Errno 28] No space left on device"),
    ("oom_killed", 137, "Killed"),
    ("oom_killed", 1, "MemoryError"),
    ("timeout", 124, "..."),
    ("sigterm", 143, "[INFO] ..."),
    ("command_not_found", 127, "bash: python3: command not found"),
    ("postgres_down", 1, "psycopg.OperationalError: connection failed: connection to server at \"127.0.0.1\", port 5432 failed: Connection refused"),
    ("python_exception", 1, "Traceback (most recent call last):\n  File \"x.py\", line 3, in <module>\nValueError: boom"),
    # 終了コード 0 でも出る警告
    ("playwright_missing", 0, "2026-10-12 07:30:02 [ERROR] scraper.jra_baba_live: Playwright含水率取得エラー: BrowserType.launch: Executable doesn't exist at /root/.cache/ms-playwright/chromium_headless_shell-1208/chrome-headless-shell-linux64/chrome-headless-shell"),
    ("parse_audit_fail", 0, "PARSE_AUDIT_FAIL: 必須項目の欠損があります（上の行を確認。docs/operations/stg_cron_schedule_20261014.md 参照）"),
    ("parse_drift_fail", 0, "PARSE_DRIFT_FAIL: flat テーブルの列で空・文字化けの急増があります"),
    ("missing_module_warn", 0, "[WARNING] src.db.etl.upsert_sires: sires UPSERT スキップ (DB 未接続等): No module named 'psycopg2'"),
    ("postgres_down_warn", 0, "[WARNING] src.db.etl.upsert_sires: sires UPSERT スキップ (DB 未接続等): (psycopg.OperationalError) connection failed"),
    ("jra_unreachable", 0, "[ERROR] scraper.jra_calendar: 9月取得失敗: HTTPSConnectionPool(host='www.jra.go.jp', port=443): Max retries exceeded"),
    ("netkeiba_unreachable", 0, "[ERROR] scraper.run: 取得失敗 [race_oikiri/202606010101]: HTTPSConnectionPool(host='race.netkeiba.com', port=443): Max retries exceeded"),
    ("netkeiba_login_failed", 0, "[ERROR] scraper.client: ログイン失敗 (nkauth cookie なし, cookies: [])"),
    ("netkeiba_400", 0, "[ERROR] scraper.run: 馬柱取得失敗 [202606010101]: 400 Client Error: Bad Request for url: https://race.netkeiba.com/race/shutuba_past.html?race_id=1"),
    ("gcs_not_saved", 0, "[WARNING] scraper.storage: GCS 未接続のため save スキップ: race_result/202106030301"),
    ("gcs_list_timeout", 0, "[WARNING] scraper.storage: batch_list_blobs 失敗 [race_result_track/2025]: Timeout of 10.0s exceeded"),
    ("permission_denied", 0, "disk_l2_weekly_access 保存失敗: [Errno 13] Permission denied: '/home/jovyan/work/keiba-vpn/data/local/meta/x.tmp'"),
]


@pytest.mark.parametrize("pid,ec,text", CASES, ids=[f"{c[0]}-{c[1]}-{i}" for i, c in enumerate(CASES)])
def test_known_failures_are_recognised(pid, ec, text):
    c = classify(ec, text)
    assert c is not None and c["pattern_id"] == pid, c
    assert c["label"] and c["hint"]
    if ec != 0:
        assert c["kind"] == "failed"


def test_kinds_by_exit_code():
    assert classify(0, "[2026-10-16T02:00:01+09:00] SKIP backfill: 優先処理(取り直し)が動いているため見送り")["kind"] == "skipped"
    assert classify(0, "PARSE_AUDIT_FAIL: x")["kind"] == "warning"
    assert classify(0, "全部うまくいった") is None
    assert classify(None, "") is None


def test_success_run_ignores_failed_only_patterns():
    assert classify(0, "Traceback (most recent call last): は文字列として出ただけ") is None
    assert classify(0, "ModuleNotFoundError: No module named 'x'")["pattern_id"] == "missing_module_warn"


def test_failed_prefers_a_failed_pattern_over_a_warning():
    text = "[ERROR] scraper.client: ログイン失敗\nTraceback (most recent call last):\nModuleNotFoundError: No module named 'bs4'"
    assert classify(1, text)["pattern_id"] == "missing_module"


def test_captures_are_put_into_the_label():
    assert "psycopg2" in classify(0, "No module named 'psycopg2'")["label"]
    assert "/data/x.tmp" in classify(0, "Permission denied: '/data/x.tmp'")["label"]


def test_unknown_failure_keeps_the_last_error_line_as_evidence():
    c = classify(2, "start\n[ERROR] something odd happened\nbye")
    assert c["pattern_id"] == "unknown" and "原因を自動判別できません" in c["label"] and c["evidence"] == "[ERROR] something odd happened"
    assert classify(9, "")["evidence"] == "終了コード 9"


def test_pattern_table_is_well_formed():
    import re
    ids = [p.id for p in PATTERNS]
    assert len(ids) == len(set(ids))
    for p in PATTERNS:
        assert p.kind in ("failed", "warning", "skipped") and p.label and p.hint
        assert p.regex or p.exit_codes, p.id
        if p.regex:
            re.compile(p.regex)
