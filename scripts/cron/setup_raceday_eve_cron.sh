#!/usr/bin/env bash
# ================================================================
# raceday-eve 前日夕方スクレイピング cron セットアップ
#
# 目的:
#   翌開催日の出馬表・馬柱・追い切りを前日 18:00 に取得し、
#   追走難度（位置取り）を出馬表のみで事前計算する。
#   （T-15 ではオッズ・馬場更新。出馬表/馬柱は取得済みならスキップ）
#
# 動作:
#   毎日 18:00 に起動 → 翌日が開催日かチェック → 非開催日は即 skip
#   開催日なら全レース: 出馬表 + 馬柱・追い切り + SmartRC → 追走難度キャッシュ
#
# Usage:
#   bash scripts/cron/setup_raceday_eve_cron.sh           # cron 追加
#   bash scripts/cron/setup_raceday_eve_cron.sh --remove  # cron 削除
#   bash scripts/cron/setup_raceday_eve_cron.sh --status  # 状態確認
# ================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
PYTHON="python3"
LOG_DIR="${PROJECT_DIR}/logs"
CRON_TAG="# KEIBA-VPN-RACEDAY-EVE"

mkdir -p "$LOG_DIR"
RUNNER="${PROJECT_DIR}/scripts/cron/run_auto_scrape_logged.sh"
chmod +x "$RUNNER" 2>/dev/null || true

remove_existing() {
    crontab -l 2>/dev/null | grep -v "$CRON_TAG" | crontab - 2>/dev/null || true
    echo "既存の KEIBA-VPN-RACEDAY-EVE cron エントリを削除しました"
}

show_status() {
    echo "=== raceday-eve cron 登録状況 ==="
    if crontab -l 2>/dev/null | grep -q "$CRON_TAG"; then
        crontab -l 2>/dev/null | grep "$CRON_TAG"
        echo ""
        echo "✓ cron 登録済み"
    else
        echo "  (未登録)"
    fi

    echo ""
    echo "=== ステータスファイル（管理画面と共有）==="
    local st="${PROJECT_DIR}/data/local/meta/auto_scrape_status.json"
    if [ -f "$st" ]; then
        ls -la "$st"
    else
        echo "  (未生成: ${st})"
    fi

    echo ""
    echo "=== 最新ログ (末尾 20 行) ==="
    local logfile="${LOG_DIR}/raceday_eve.log"
    if [ -f "$logfile" ]; then
        tail -20 "$logfile"
    else
        echo "  (ログファイルなし: ${logfile})"
    fi
}

if [ "${1:-}" = "--remove" ]; then
    remove_existing
    echo "完了"
    exit 0
fi

if [ "${1:-}" = "--status" ]; then
    show_status
    exit 0
fi

# 既存エントリを除去してから追加
remove_existing

CRON_ENTRIES=$(cat <<EOF

# =========================================================
# keiba-vpn raceday-eve: 前日夕方 出馬表+馬柱・追い切り+追走難度 ${CRON_TAG}
#   毎日 18:00 に起動 → 翌日が開催日のときのみ取得
#   非開催日は数秒で skip して終了
# =========================================================
# crontab の 5 フィールドは CRON_TZ に従う（未設定だとサーバ TZ＝UTC になりがち）
CRON_TZ=Asia/Tokyo

# --- 毎日 18:00 に raceday-eve タスクを実行（ログは開始・終了行で区切る）---
0 18 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${RUNNER} ${PROJECT_DIR} raceday-eve logs/raceday_eve.log ${CRON_TAG}

# --- 毎週金曜 18:00 — 馬名リスト + 成長曲線（calculated_data 一括更新）---
0 18 * * 5 cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${RUNNER} ${PROJECT_DIR} horse-name-index logs/horse_name_index.log ${CRON_TAG}-HORSE-INDEX

EOF
)

(crontab -l 2>/dev/null; echo "$CRON_ENTRIES") | crontab -

echo "=== raceday-eve cron 設定完了 ==="
echo ""
echo "スケジュール:"
echo "  毎日 18:00 — raceday-eve（翌開催日のみ 出馬表・馬柱・追い切り + 追走難度）"
echo "  毎週金曜 18:00 — horse-name-index（馬名リスト + 成長曲線 → calculated_data）"
echo ""
echo "ログ:"
echo "  ${LOG_DIR}/raceday_eve.log"
echo "  ${LOG_DIR}/horse_name_index.log"
echo ""
echo "操作コマンド:"
echo "  cron 確認: crontab -l | grep KEIBA-VPN-RACEDAY-EVE"
echo "  cron 削除: bash ${PROJECT_DIR}/scripts/cron/setup_raceday_eve_cron.sh --remove"
echo "  手動実行:  cd ${PROJECT_DIR} && ${PYTHON} -m src.scraper.auto_scrape --task raceday-eve"
echo "  管理画面: /cron-jobs （最終実行は data/local/meta/auto_scrape_status.json と同期）"
