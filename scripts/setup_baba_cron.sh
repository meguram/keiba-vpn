#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  JRA馬場情報 スケジュール連動デーモン セットアップ
# ═══════════════════════════════════════════════════════════════
#
# JRA年間開催スケジュール (公式カレンダーAPI) から
# 全開催日 + 前日発表日を自動判定し、適切な時間帯にポーリングする。
#
# 変則開催 (祝日月曜・火曜等) も完全にカバー。
#
# Usage:
#   bash scripts/setup_baba_cron.sh              # systemd ユーザーサービス登録 (推奨)
#   bash scripts/setup_baba_cron.sh --remove     # 削除
#   bash scripts/setup_baba_cron.sh --status     # 状態確認
#   bash scripts/setup_baba_cron.sh --schedule   # 直近のスケジュール表示

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$(which python3)"
LOG_DIR="${SCRIPT_DIR}/logs"
SERVICE_NAME="jra-baba-watch"
CRON_TAG="# JRA-BABA-LIVE"

mkdir -p "$LOG_DIR"

# ── systemd ユーザーサービス ──

setup_systemd() {
    local unit_dir="$HOME/.config/systemd/user"
    mkdir -p "$unit_dir"

    cat > "$unit_dir/${SERVICE_NAME}.service" << UNIT
[Unit]
Description=JRA Baba Live Watcher - 馬場情報スケジュール連動デーモン
After=network-online.target

[Service]
Type=simple
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${PYTHON} -m scraper.jra_baba_live --watch
Restart=on-failure
RestartSec=60
StandardOutput=append:${LOG_DIR}/baba_watch.log
StandardError=append:${LOG_DIR}/baba_watch.log

[Install]
WantedBy=default.target
UNIT

    systemctl --user daemon-reload
    systemctl --user enable "${SERVICE_NAME}.service"
    systemctl --user start "${SERVICE_NAME}.service"

    echo "✓ systemd ユーザーサービス登録完了"
    echo ""
    echo "  JRA年間スケジュールに連動して自動でポーリング"
    echo "  変則開催 (祝日月曜・火曜等) も完全カバー"
    echo ""
    echo "  サービス: ${SERVICE_NAME}"
    echo "  ログ:     ${LOG_DIR}/baba_watch.log"
    echo ""
    echo "操作:"
    echo "  状態確認:  systemctl --user status ${SERVICE_NAME}"
    echo "  ログ:      journalctl --user -u ${SERVICE_NAME} -f"
    echo "  停止:      systemctl --user stop ${SERVICE_NAME}"
    echo "  再起動:    systemctl --user restart ${SERVICE_NAME}"
}

# ── 削除 ──

remove_all() {
    if systemctl --user is-active "${SERVICE_NAME}" &>/dev/null; then
        systemctl --user stop "${SERVICE_NAME}"
        systemctl --user disable "${SERVICE_NAME}"
        echo "✓ systemd サービス停止・無効化"
    fi
    rm -f "$HOME/.config/systemd/user/${SERVICE_NAME}.service"
    systemctl --user daemon-reload 2>/dev/null || true

    if crontab -l 2>/dev/null | grep -q "${CRON_TAG}"; then
        crontab -l 2>/dev/null | grep -v "${CRON_TAG}" | crontab -
        echo "✓ レガシー cron ジョブ削除"
    fi

    echo "✓ 全て削除完了"
}

# ── 状態確認 ──

show_status() {
    echo "=== systemd サービス ==="
    if systemctl --user is-active "${SERVICE_NAME}" &>/dev/null; then
        systemctl --user status "${SERVICE_NAME}" --no-pager 2>/dev/null | head -15
    else
        echo "  (未登録 or 停止中)"
    fi

    echo ""
    echo "=== 最新ログ (末尾15行) ==="
    if [ -f "${LOG_DIR}/baba_watch.log" ]; then
        tail -15 "${LOG_DIR}/baba_watch.log"
    else
        echo "  (ログファイルなし)"
    fi
}

# ── スケジュール表示 ──

show_schedule() {
    cd "${SCRIPT_DIR}"
    ${PYTHON} -m scraper.jra_baba_live --schedule --schedule-days "${2:-30}"
}

# ── メイン ──

case "${1:-}" in
    --remove)
        remove_all
        ;;
    --status)
        show_status
        ;;
    --schedule)
        show_schedule "$@"
        ;;
    *)
        setup_systemd
        ;;
esac
