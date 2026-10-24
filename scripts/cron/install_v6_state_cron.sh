#!/usr/bin/env bash
# v6 の推論用の参照データの更新を crontab に登録する（冪等。タグ KEIBA-V6-STATE の行だけを入れ替える）。ホストで root として実行する。
# 登録前に crontab -l を data/local/meta/cron_backups/ へ退避する。
#   rb-state-daily  … RB の履歴状態を日次で作り直す（毎日 23:20 UTC = 08:20 JST。22:00 UTC の export_tables_daily と 22:40 UTC の feature-accumulate の後。
#                      スクリプト内でも export のロックを待つ。十数秒・軽い）
#   px-ridge-window … 新しい半年（1/1・7/1）の 30 日前から、PX のリッジを作る（毎週日曜 18:30 UTC = 月曜 03:30 JST。作るのは窓の前だけで、普段は何もしない。heavy_run 経由・数分）
# Usage: bash scripts/cron/install_v6_state_cron.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-V6-STATE"
RB="20 23 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh rb-state-daily -- bash ${PROJECT_DIR}/scripts/cron/rb_state_daily.sh ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/rb_state_daily.cron.log 2>&1 # ${TAG}"
PX="30 18 * * 0 cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh px-ridge-window -- bash ${PROJECT_DIR}/scripts/cron/px_ridge_window.sh ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/px_ridge_window.cron.log 2>&1 # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data/local/meta/cron_backups"
BK="${PROJECT_DIR}/data/local/meta/cron_backups/crontab_before_v6_state_$(date +%Y%m%d_%H%M%S).txt"
crontab -l 2>/dev/null > "$BK" || true
echo "crontab を退避しました: $BK"
TMP="$(mktemp)"
grep -v "# ${TAG}" "$BK" > "$TMP" || true
echo "$RB" >> "$TMP"
echo "$PX" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "登録しました:"; echo "$RB"; echo "$PX"
