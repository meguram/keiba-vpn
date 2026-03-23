#!/usr/bin/env bash
# ================================================================
# keiba-vpn 全サービス監視 cron セットアップ
#
# 監視対象:
#   - keiba-vpn FastAPI サーバー (port 8000)
#   - MLflow Tracking サーバー   (port 5000)
#
# 実行すると以下を crontab に追加する:
#   - 3分ごとのヘルスチェック & 自動再起動
#   - システム起動時の全サービス自動起動
#   - 週次のログローテーション
#
# Usage:
#   bash scripts/setup_watchdog_cron.sh          # 設定追加
#   bash scripts/setup_watchdog_cron.sh --remove  # 設定削除
# ================================================================

set -euo pipefail

PROJECT_DIR="/home/hirokiakataoka/project/myproject/keiba-vpn"
WATCHDOG="${PROJECT_DIR}/scripts/server_watchdog.sh"
CRON_TAG="# KEIBA-VPN-WATCHDOG"

remove_existing() {
    crontab -l 2>/dev/null | grep -v "$CRON_TAG" | crontab - 2>/dev/null || true
    echo "既存の KEIBA-VPN-WATCHDOG cron エントリを削除しました"
}

if [ "${1:-}" = "--remove" ]; then
    remove_existing
    echo "完了"
    exit 0
fi

# 既存エントリを除去してから追加
remove_existing

CRON_ENTRIES=$(cat <<EOF

# =========================================================
# keiba-vpn 全サービス監視 ${CRON_TAG}
#   API (port 8000) + MLflow (port 5000)
#   3分間隔ヘルスチェック + ダウン時自動再起動
# =========================================================

# --- 3分ごとに全サービスをヘルスチェック → ダウン時は自動再起動 ---
*/3 * * * * ${WATCHDOG} ${CRON_TAG}

# --- システム起動時に全サービスを起動 ---
@reboot sleep 15 && ${WATCHDOG} ${CRON_TAG}

# --- 週次ログローテーション (月曜 AM 0:00) ---
0 0 * * 1 find ${PROJECT_DIR}/logs -name "*.log" -size +50M -exec truncate -s 0 {} \; ${CRON_TAG}

EOF
)

# crontab に追記
(crontab -l 2>/dev/null; echo "$CRON_ENTRIES") | crontab -

echo "=== keiba-vpn 全サービス watchdog cron 設定完了 ==="
echo ""
echo "監視対象:"
echo "  - keiba-vpn API  : http://127.0.0.1:8000/api/health"
echo "  - MLflow Tracking: http://127.0.0.1:5000/health"
echo ""
echo "追加されたエントリ:"
echo "  - */3 * * * *  : 3分ごとのヘルスチェック & 自動再起動"
echo "  - @reboot      : システム起動時に全サービスを自動起動"
echo "  - 0 0 * * 1    : 週次ログローテーション"
echo ""
echo "操作コマンド:"
echo "  状態確認: ${WATCHDOG} --status"
echo "  cron確認: crontab -l | grep KEIBA-VPN-WATCHDOG"
echo "  cron削除: bash ${PROJECT_DIR}/scripts/setup_watchdog_cron.sh --remove"
