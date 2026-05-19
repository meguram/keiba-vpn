#!/usr/bin/env bash
# ================================================================
# raceday-eve 前日夕方スクレイピング cron セットアップ
#
# 目的:
#   翌開催日の馬柱・調教 / 追い切りデータを、レース直前ではなく
#   前日 18:00 に先行取得する。
#   （発走 T-15 バンドルでは skip_existing=True のフォールバック扱い）
#
# 動作:
#   毎日 18:00 に起動 → 翌日が開催日かチェック → 非開催日は即 skip
#   開催日なら全レース分の 馬柱・追い切り を skip_existing=False で取得
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
# keiba-vpn raceday-eve: 前日夕方 馬柱・追い切り先行取得 ${CRON_TAG}
#   毎日 18:00 に起動 → 翌日が開催日のときのみ取得
#   非開催日は数秒で skip して終了
# =========================================================

# --- 毎日 18:00 に raceday-eve タスクを実行 ---
0 18 * * * cd ${PROJECT_DIR} && ${PYTHON} -m src.scraper.auto_scrape --task raceday-eve >> ${LOG_DIR}/raceday_eve.log 2>&1 ${CRON_TAG}

EOF
)

(crontab -l 2>/dev/null; echo "$CRON_ENTRIES") | crontab -

echo "=== raceday-eve cron 設定完了 ==="
echo ""
echo "スケジュール: 毎日 18:00"
echo "  翌日が開催日 → 全レースの馬柱・追い切りを取得"
echo "  翌日が非開催日 → 即 skip して終了"
echo ""
echo "ログ: ${LOG_DIR}/raceday_eve.log"
echo ""
echo "操作コマンド:"
echo "  cron 確認: crontab -l | grep KEIBA-VPN-RACEDAY-EVE"
echo "  cron 削除: bash ${PROJECT_DIR}/scripts/cron/setup_raceday_eve_cron.sh --remove"
echo "  手動実行:  cd ${PROJECT_DIR} && ${PYTHON} -m src.scraper.auto_scrape --task raceday-eve"
