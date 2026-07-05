#!/usr/bin/env bash
# =============================================================================
# keiba-vpn 全 cron ジョブ 一括セットアップ
#
# SLA マッピング (docs/requirements/data/scrape_process.md):
#   SLA 0 → daily-race-lists   (07:00 JST & 17:00 JST)
#   SLA 1 → raceday-eve        (18:00 JST 前日)
#   SLA 2 → jra-baba-morning   (05:00–09:00 JST 毎10分)
#   SLA 3 → raceday-runner     (07:30 JST 常駐, T-15 各R)
#   SLA 4 → raceday-result-runner (07:30 JST 常駐, T+15 各R)
#   SLA 5 → raceday-evening    (17:30 JST)
#   SLA 6 → weekly-update      (17:30 JST 金曜)
#   SLA 7 → raceday-eve 連動 (horse_result 等は raceday-eve 内で取得)
#
# 注意: システム TZ は UTC。スケジュール時刻はすべて UTC 換算で記述する（JST = UTC+9）。
#
# Usage:
#   bash scripts/cron/setup_all_cron.sh            # 設定プレビュー (show)
#   bash scripts/cron/setup_all_cron.sh install    # crontab に登録
#   bash scripts/cron/setup_all_cron.sh remove     # 全 keiba-vpn エントリ削除
#   bash scripts/cron/setup_all_cron.sh status     # crontab + タスク状態
#   bash scripts/cron/setup_all_cron.sh test       # ドライラン検証
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"

# Python を自動解決（conda 優先、ハードコード不要）
if [ -x "/opt/conda/bin/python3" ]; then
    PYTHON="/opt/conda/bin/python3"
elif [ -x "$(command -v python3 2>/dev/null)" ]; then
    PYTHON="$(command -v python3)"
else
    PYTHON="python3"
fi
LOG_DIR="${PROJECT_DIR}/logs"
RUNNER="${PROJECT_DIR}/scripts/cron/run_auto_scrape_logged.sh"
WATCHDOG="${PROJECT_DIR}/scripts/server/server_watchdog.sh"
UPDATE_JT="${PROJECT_DIR}/scripts/cron/update_jockey_trainer_stats.sh"
ROTATE_LOGS="${PROJECT_DIR}/scripts/cron/rotate_logs.sh"
GIT_PULL="${PROJECT_DIR}/scripts/cron/git_pull_hourly.sh"

mkdir -p "$LOG_DIR"
chmod +x "$RUNNER"     2>/dev/null || true
chmod +x "$WATCHDOG"   2>/dev/null || true
chmod +x "$UPDATE_JT"  2>/dev/null || true
chmod +x "$ROTATE_LOGS" 2>/dev/null || true
chmod +x "$GIT_PULL"   2>/dev/null || true

# -------------------------------------------------------------------
# 削除対象タグ (全 keiba-vpn cron エントリをまとめて除去)
# -------------------------------------------------------------------
ALL_TAGS=(
    "KEIBA-VPN-RACEDAY-EVE"
    "KEIBA-VPN-WATCHDOG"
    "KEIBA_BACKFILL"
    "KEIBA_JT_STATS"
    "KEIBA_GIT_PULL"
    "KEIBA-VPN-ALL"
)

remove_all() {
    local current
    current="$(crontab -l 2>/dev/null || true)"
    local cleaned="$current"
    for tag in "${ALL_TAGS[@]}"; do
        cleaned="$(echo "$cleaned" | grep -v "$tag" || true)"
    done
    # CRON_TZ 行も除去（再追加するので）
    cleaned="$(echo "$cleaned" | grep -v '^CRON_TZ=' || true)"
    # 旧 setup_raceday_eve_cron.sh が残したタグなし旧コメント行を除去
    cleaned="$(echo "$cleaned" | grep -v -E \
        '^# ={5,}|^#\s*(毎日|毎週金曜) [0-9]+:[0-9]+|^#\s*非開催日|^# ---' || true)"
    # 連続空行を1行に圧縮してから登録
    echo "$cleaned" | cat -s | crontab -
    echo "既存の keiba-vpn cron エントリをすべて削除しました"
}

# -------------------------------------------------------------------
# cron エントリ生成
# -------------------------------------------------------------------
generate_cron() {
    cat <<CRON_EOF
# =================================================================
# keiba-vpn 全自動スクレイピング cron (setup_all_cron.sh で管理)
#
# !! このブロックを手動編集しないこと。
#    変更は setup_all_cron.sh を修正して install し直すこと。
#
# 時刻はすべて UTC で記述（JST = UTC+9）。
# CRON_TZ はこの環境の Debian vixie-cron では無効のため使用しない。
# =================================================================
# Python / conda PATH を明示（cron は通常 PATH が限定的なため）
PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin

# ── サービス監視 (Watchdog) ─── UTC: 常時 / JST: 常時 ─────────── # KEIBA-VPN-ALL
*/3 * * * * ${WATCHDOG} # KEIBA-VPN-WATCHDOG
@reboot sleep 15 && ${WATCHDOG} # KEIBA-VPN-WATCHDOG

# ── ログローテーション ─────────── UTC: 19:30 / JST: 04:30 ──────── # KEIBA-VPN-ALL
30 19 * * * TZ=Asia/Tokyo bash ${ROTATE_LOGS} >> ${LOG_DIR}/rotate_logs.log 2>&1 # KEIBA-VPN-WATCHDOG

# ── git pull（10 分ごと） ──────── UTC: */10 / JST: */10 ─────────── # KEIBA-VPN-ALL
*/10 * * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${GIT_PULL} # KEIBA_GIT_PULL

# ── SLA 0: 毎日 レース一覧 朝取り ─ UTC: 22:00 / JST: 07:00 ──── # KEIBA-VPN-ALL
0 22 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${RUNNER} ${PROJECT_DIR} daily-race-lists logs/daily_race_lists_am.log # KEIBA-VPN-ALL
# ── SLA 0: 毎日 レース一覧 夕方 ─── UTC: 08:00 / JST: 17:00 ──── # KEIBA-VPN-ALL
0 8 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${RUNNER} ${PROJECT_DIR} daily-race-lists logs/daily_race_lists_pm.log # KEIBA-VPN-ALL

# ── SLA 2: JRA馬場情報 朝ポーリング UTC: 20:00-23:50 / JST: 05:00-08:50 # KEIBA-VPN-ALL
# 開催日以外は自動スキップ
*/10 20-23 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${RUNNER} ${PROJECT_DIR} jra-baba-morning logs/jra_baba_morning.log # KEIBA-VPN-ALL

# ── 騎手・調教師統計 ─────────── UTC: 20:30 / JST: 05:30 ────────── # KEIBA-VPN-ALL
30 20 * * * ${UPDATE_JT} # KEIBA_JT_STATS

# ── SLA 3: 開催日ランナー ───────── UTC: 22:30 / JST: 07:30 ──────── # KEIBA-VPN-ALL
# 常駐プロセス: 全レース終了まで内部ループで待機・実行
# 非開催日は起動後数秒でスキップ終了
30 22 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${RUNNER} ${PROJECT_DIR} raceday-runner logs/raceday_runner.log # KEIBA-VPN-ALL

# ── SLA 4: 速報結果ランナー ──────── UTC: 22:30 / JST: 07:30 ──────── # KEIBA-VPN-ALL
30 22 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${RUNNER} ${PROJECT_DIR} raceday-result-runner logs/raceday_result_runner.log # KEIBA-VPN-ALL

# ── SLA 5: 開催日夕方 速報まとめ ── UTC: 08:30 / JST: 17:30 ──────── # KEIBA-VPN-ALL
30 8 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${RUNNER} ${PROJECT_DIR} raceday-evening logs/raceday_evening.log # KEIBA-VPN-ALL

# ── SLA 6: 金曜週次更新 ─────────── UTC: 08:30 Fri / JST: 17:30 Fri # KEIBA-VPN-ALL
30 8 * * 5 cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${RUNNER} ${PROJECT_DIR} weekly-update logs/weekly_update.log # KEIBA-VPN-ALL

# ── SLA 1: 前日夕方 出馬表・馬柱 ── UTC: 09:00 / JST: 18:00 ──────── # KEIBA-VPN-ALL
0 9 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${RUNNER} ${PROJECT_DIR} raceday-eve logs/raceday_eve.log # KEIBA-VPN-RACEDAY-EVE

# ── 金曜 馬名インデックス ─────── UTC: 09:00 Fri / JST: 18:00 Fri ─── # KEIBA-VPN-ALL
0 9 * * 5 cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${RUNNER} ${PROJECT_DIR} horse-name-index logs/horse_name_index.log # KEIBA-VPN-RACEDAY-EVE

# ── 過去データ Backfill (深夜) ────────────────────────────────── # KEIBA-VPN-ALL
# Phase fast: レース結果+出馬表
# UTC: 15:00=JST:00:00 / 16:00=JST:01:00 / 17:00=JST:02:00 / 18:00=JST:03:00 / 19:00=JST:04:00
0 15 * * *   cd ${PROJECT_DIR} && TZ=Asia/Tokyo ${PYTHON} -m src.scraper.backfill --year 2026 --phase fast --max-dates 7 >> ${LOG_DIR}/backfill_2026.log 2>&1 # KEIBA_BACKFILL
0 16 * * *   cd ${PROJECT_DIR} && TZ=Asia/Tokyo ${PYTHON} -m src.scraper.backfill --year 2025 --phase fast --max-dates 5 >> ${LOG_DIR}/backfill_2025.log 2>&1 # KEIBA_BACKFILL
0 17 * * *   cd ${PROJECT_DIR} && TZ=Asia/Tokyo ${PYTHON} -m src.scraper.backfill --year 2024 --phase fast --max-dates 5 >> ${LOG_DIR}/backfill_2024.log 2>&1 # KEIBA_BACKFILL
0 18 * * *   cd ${PROJECT_DIR} && TZ=Asia/Tokyo ${PYTHON} -m src.scraper.backfill --year 2023 --phase fast --max-dates 5 >> ${LOG_DIR}/backfill_2023.log 2>&1 # KEIBA_BACKFILL
0 19 * * *   cd ${PROJECT_DIR} && TZ=Asia/Tokyo ${PYTHON} -m src.scraper.backfill --year 2022 --phase fast --max-dates 5 >> ${LOG_DIR}/backfill_2022.log 2>&1 # KEIBA_BACKFILL
# Phase horse: 馬情報一括   UTC: 21:00 / JST: 06:00
0 21 * * *   cd ${PROJECT_DIR} && TZ=Asia/Tokyo ${PYTHON} -m src.scraper.backfill --phase horse >> ${LOG_DIR}/backfill_horse.log 2>&1 # KEIBA_BACKFILL
# Phase full: 補助データ (race_result_on_time 含む)  UTC: 22:30=JST:07:30 / 23:00=JST:08:00 / 00:00=JST:09:00
30 22 * * *  cd ${PROJECT_DIR} && TZ=Asia/Tokyo ${PYTHON} -m src.scraper.backfill --year 2026 --phase full --max-dates 5 >> ${LOG_DIR}/backfill_full_2026.log 2>&1 # KEIBA_BACKFILL
0 23 * * *   cd ${PROJECT_DIR} && TZ=Asia/Tokyo ${PYTHON} -m src.scraper.backfill --year 2025 --phase full --max-dates 3 >> ${LOG_DIR}/backfill_full_2025.log 2>&1 # KEIBA_BACKFILL
0  0 * * *   cd ${PROJECT_DIR} && TZ=Asia/Tokyo ${PYTHON} -m src.scraper.backfill --year 2024 --phase full --max-dates 3 >> ${LOG_DIR}/backfill_full_2024.log 2>&1 # KEIBA_BACKFILL
# 古い年度: 週2回  UTC: 17:00=JST:02:00(月木) / 18:00=JST:03:00(火金)
0 17 * * 1,4 cd ${PROJECT_DIR} && TZ=Asia/Tokyo ${PYTHON} -m src.scraper.backfill --year 2021 --phase fast --max-dates 5 >> ${LOG_DIR}/backfill_2021.log 2>&1 # KEIBA_BACKFILL
0 18 * * 2,5 cd ${PROJECT_DIR} && TZ=Asia/Tokyo ${PYTHON} -m src.scraper.backfill --year 2020 --phase fast --max-dates 5 >> ${LOG_DIR}/backfill_2020.log 2>&1 # KEIBA_BACKFILL
# backfill ログは rotate_logs.sh (04:30 JST) で一括管理 # KEIBA_BACKFILL

CRON_EOF
}

# -------------------------------------------------------------------
# サブコマンド: show
# -------------------------------------------------------------------
cmd_show() {
    echo "============================================================"
    echo " keiba-vpn cron 設定プレビュー (システム TZ: $(cat /etc/timezone 2>/dev/null || date +%Z))"
    echo "============================================================"
    echo ""
    echo "以下のエントリが crontab に登録されます:"
    echo ""
    generate_cron
    echo ""
    echo "登録: bash $0 install"
    echo "削除: bash $0 remove"
}

# -------------------------------------------------------------------
# サブコマンド: install
# -------------------------------------------------------------------
cmd_install() {
    echo "=== keiba-vpn cron 全ジョブ インストール ==="
    echo "システム TZ: $(cat /etc/timezone 2>/dev/null || date +%Z)"
    echo "スケジュール: UTC 時刻で記述 (JST=UTC+9。TZ=Asia/Tokyo は各コマンドに付与)"
    echo ""

    # 既存エントリを除去
    remove_all

    # 新エントリを追加
    local new_entries
    new_entries="$(generate_cron)"
    {
        crontab -l 2>/dev/null || true
        echo ""
        echo "$new_entries"
    } | crontab -

    echo ""
    echo "✅ crontab へのインストール完了"
    echo ""
    echo "=== 登録確認 ==="
    crontab -l | grep -E "KEIBA|^PATH"
    echo ""
    echo "=== スケジュール概要 (JST) ==="
    echo "  05:00–09:00 毎10分: jra-baba-morning (開催日のみ)"
    echo "  05:30       毎日:   騎手・調教師統計再生成"
    echo "  07:00       毎日:   daily-race-lists (朝取り)"
    echo "  07:30       毎日:   raceday-runner + raceday-result-runner (開催日のみ常駐)"
    echo "  17:00       毎日:   daily-race-lists (夕方更新)"
    echo "  17:30       毎日:   raceday-evening (開催日のみ)"
    echo "  17:30       金曜:   weekly-update"
    echo "  18:00       毎日:   raceday-eve (翌開催日のみ)"
    echo "  18:00       金曜:   horse-name-index"
    echo "  01:00–09:00 深夜:   backfill (年度別)"
    echo "  */3         常時:   watchdog (API + MLflow)"
    echo "  */10        常時:   git pull（リポジトリ最新化）"
    echo ""
    echo "ログディレクトリ: ${LOG_DIR}/"
}

# -------------------------------------------------------------------
# サブコマンド: remove
# -------------------------------------------------------------------
cmd_remove() {
    remove_all
    echo "✅ keiba-vpn cron エントリをすべて削除しました"
    echo "確認: crontab -l"
}

# -------------------------------------------------------------------
# サブコマンド: status
# -------------------------------------------------------------------
cmd_status() {
    echo "=== crontab 登録状況 ==="
    if crontab -l 2>/dev/null | grep -q "KEIBA"; then
        echo "✓ keiba-vpn エントリあり"
        crontab -l | grep -E "KEIBA|^PATH"
    else
        echo "  (未登録)"
    fi

    echo ""
    echo "=== タスク最終実行状態 ==="
    cd "$PROJECT_DIR" && $PYTHON -m src.scraper.auto_scrape --status 2>/dev/null || echo "(auto_scrape status 取得失敗)"

    echo ""
    echo "=== ログサイズ ==="
    for f in "${LOG_DIR}"/*.log; do
        [ -f "$f" ] || continue
        printf "  %-45s %s\n" "$(basename "$f")" "$(du -h "$f" 2>/dev/null | cut -f1)"
    done
}

# -------------------------------------------------------------------
# サブコマンド: test — 各タスクのドライラン検証
# -------------------------------------------------------------------
cmd_test() {
    echo "============================================================"
    echo " keiba-vpn cron テスト (ドライラン)"
    echo " システム TZ: $(cat /etc/timezone 2>/dev/null || date +%Z)"
    echo " JST 現在時刻: $(TZ=Asia/Tokyo date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "============================================================"
    echo ""

    cd "$PROJECT_DIR"
    PASS=0; FAIL=0

    run_test() {
        local name="$1"
        local cmd="$2"
        printf "  %-30s ... " "$name"
        local ec=0
        eval "$cmd" > /tmp/keiba_cron_test_$$.log 2>&1 || ec=$?
        # SIGPIPE (141) はパイプ正常終了とみなす
        if [ "$ec" -eq 0 ] || [ "$ec" -eq 141 ]; then
            echo "OK"
            PASS=$((PASS+1))
        else
            echo "FAIL (exit $ec)"
            tail -3 /tmp/keiba_cron_test_$$.log | sed 's/^/    /'
            FAIL=$((FAIL+1))
        fi
        rm -f /tmp/keiba_cron_test_$$.log
    }

    echo "[1/5] Python モジュール読み込み確認"
    run_test "auto_scrape import" \
        "TZ=Asia/Tokyo $PYTHON -c 'from src.scraper.auto_scrape import TASKS; assert len(TASKS) >= 8'"
    run_test "backfill import" \
        "TZ=Asia/Tokyo $PYTHON -c 'from src.scraper import backfill'"
    run_test "jra_baba_live import" \
        "TZ=Asia/Tokyo $PYTHON -c 'from src.scraper.jra_baba_live import run_cron_job'"

    echo ""
    echo "[2/5] JST タイムゾーン動作確認"
    run_test "TZ=Asia/Tokyo date" \
        "TZ=Asia/Tokyo $PYTHON -c \"
import datetime, zoneinfo
jst = zoneinfo.ZoneInfo('Asia/Tokyo')
now = datetime.datetime.now(jst)
print(f'JST: {now.strftime(\\\"%Y-%m-%d %H:%M:%S %Z\\\")}')
assert now.tzinfo is not None
\""
    run_test "PATH 設定確認" \
        "grep -q 'PATH=/opt/conda' <(bash $0 show)"

    echo ""
    echo "[3/5] run_auto_scrape_logged.sh 動作確認"
    run_test "runner スクリプト存在" \
        "test -x '$RUNNER'"
    run_test "daily-race-lists (skip可)" \
        "TZ=Asia/Tokyo timeout 30 bash $RUNNER $PROJECT_DIR daily-race-lists /tmp/test_daily_race_lists_$$.log"
    rm -f "/tmp/test_daily_race_lists_$$.log"

    echo ""
    echo "[4/5] jra-baba-morning (非開催日スキップ確認)"
    run_test "jra-baba-morning (skip OK)" \
        "TZ=Asia/Tokyo timeout 30 $PYTHON -m src.scraper.auto_scrape --task jra-baba-morning"

    echo ""
    echo "[5/5] crontab 生成内容検証"
    run_test "raceday-runner が UTC 22:30 に設定されている" \
        "bash $0 show | grep -q '30 22.*raceday-runner'"
    run_test "全 SLA タスクが含まれる (raceday-runner)" \
        "bash $0 show | grep -q 'raceday-runner'"
    run_test "全 SLA タスクが含まれる (raceday-result-runner)" \
        "bash $0 show | grep -q 'raceday-result-runner'"
    run_test "全 SLA タスクが含まれる (raceday-evening)" \
        "bash $0 show | grep -q 'raceday-evening'"
    run_test "全 SLA タスクが含まれる (weekly-update)" \
        "bash $0 show | grep -q 'weekly-update'"
    run_test "全 SLA タスクが含まれる (daily-race-lists)" \
        "bash $0 show | grep -q 'daily-race-lists'"
    run_test "全 SLA タスクが含まれる (jra-baba-morning)" \
        "bash $0 show | grep -q 'jra-baba-morning'"
    run_test "全 SLA タスクが含まれる (jt_stats)" \
        "bash $0 show | grep -q 'update_jockey_trainer_stats'"
    run_test "backfill エントリが含まれる" \
        "bash $0 show | grep -q 'backfill'"
    run_test "watchdog エントリが含まれる" \
        "bash $0 show | grep -q 'server_watchdog'"
    run_test "git pull エントリが含まれる" \
        "bash $0 show | grep -q 'git_pull_hourly'"

    echo ""
    echo "============================================================"
    echo " テスト結果: PASS=${PASS}  FAIL=${FAIL}"
    echo "============================================================"
    if [ "$FAIL" -eq 0 ]; then
        echo "✅ 全テスト通過。本番登録: bash $0 install"
        return 0
    else
        echo "❌ ${FAIL} 件のテストが失敗しました。上記ログを確認してください。"
        return 1
    fi
}

# -------------------------------------------------------------------
# メイン
# -------------------------------------------------------------------
case "${1:-show}" in
    show)    cmd_show    ;;
    install) cmd_install ;;
    remove)  cmd_remove  ;;
    status)  cmd_status  ;;
    test)    cmd_test    ;;
    *)
        echo "Usage: $0 {show|install|remove|status|test}"
        exit 1
        ;;
esac
