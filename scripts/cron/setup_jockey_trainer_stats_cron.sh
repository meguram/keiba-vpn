#!/usr/bin/env bash
# =============================================================================
# 騎手・調教師統計（jt_race_features / lookup_*）の cron 登録
#
# データ取得（export_tables 等）の後に走らせる想定。既定は毎日 05:30。
#
# Usage:
#   bash scripts/cron/setup_jockey_trainer_stats_cron.sh           # 表示のみ
#   bash scripts/cron/setup_jockey_trainer_stats_cron.sh install   # crontab に追加
#   bash scripts/cron/setup_jockey_trainer_stats_cron.sh remove    # 削除
#   bash scripts/cron/setup_jockey_trainer_stats_cron.sh status    # 直近ログ・manifest
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
UPDATE_SH="${PROJECT_DIR}/scripts/cron/update_jockey_trainer_stats.sh"
LOG_DIR="${PROJECT_DIR}/logs"
MANIFEST="${PROJECT_DIR}/data/features/jockey_trainer_stats/_manifest.json"

CRON_TAG="# KEIBA_JT_STATS"

# 分 時 日 月 曜 — サーバの TZ に依存（必要なら crontab 先頭に CRON_TZ=Asia/Tokyo）
CRON_SCHEDULE="${JT_STATS_CRON_SCHEDULE:-30 5 * * *}"

mkdir -p "$LOG_DIR"
chmod +x "$UPDATE_SH" 2>/dev/null || true

generate_cron() {
  cat <<CRON_ENTRIES
# =========================================================
# keiba-vpn: 騎手・調教師成績統計の再生成 ${CRON_TAG}
#   src.pipeline.build_jockey_trainer_stats → data/features/jockey_trainer_stats/
#   環境変数 JT_STATS_CRON_SCHEDULE でスケジュール変更可（既定 30 5 * * *）
# =========================================================
${CRON_SCHEDULE} ${UPDATE_SH} ${CRON_TAG}

CRON_ENTRIES
}

case "${1:-show}" in
  show)
    echo "プロジェクト: ${PROJECT_DIR}"
    echo "実行スクリプト: ${UPDATE_SH}"
    echo "スケジュール（JT_STATS_CRON_SCHEDULE 未設定時）: ${CRON_SCHEDULE}"
    echo ""
    echo "以下の 1 行が crontab に追加されます:"
    echo ""
    generate_cron
    echo ""
    echo "登録: bash $0 install"
    echo "地方も含める場合は ${UPDATE_SH} を --nar 付きに編集するか、手動で:"
    echo "  ${CRON_SCHEDULE} ${UPDATE_SH} --nar ${CRON_TAG}"
    ;;

  install)
    if [ ! -x "$UPDATE_SH" ]; then
      chmod +x "$UPDATE_SH"
    fi
    existing=$(crontab -l 2>/dev/null || true)
    cleaned=$(echo "$existing" | grep -v "$CRON_TAG" || true)

    new_entries=$(generate_cron)
    {
      echo "$cleaned"
      echo ""
      echo "$new_entries"
    } | crontab -

    echo "✅ KEIBA_JT_STATS cron を登録しました"
    echo "確認: crontab -l | grep KEIBA_JT_STATS"
    echo "ログ: tail -f ${LOG_DIR}/jockey_trainer_stats.log"
    ;;

  remove)
    existing=$(crontab -l 2>/dev/null || true)
    cleaned=$(echo "$existing" | grep -v "$CRON_TAG" || true)
    echo "$cleaned" | crontab -
    echo "✅ KEIBA_JT_STATS cron を削除しました"
    ;;

  status)
    echo "=== merge spec (_merge_spec.json) ==="
    MS="${PROJECT_DIR}/data/features/jockey_trainer_stats/_merge_spec.json"
    if [ -f "$MS" ]; then
      head -n 35 "$MS"
    else
      echo "(未生成) $MS"
    fi
    echo ""
    echo "=== manifest ==="
    if [ -f "$MANIFEST" ]; then
      head -n 25 "$MANIFEST"
    else
      echo "(未生成) $MANIFEST"
    fi
    echo ""
    echo "=== ログ末尾 (${LOG_DIR}/jockey_trainer_stats.log) ==="
    if [ -f "${LOG_DIR}/jockey_trainer_stats.log" ]; then
      tail -n 40 "${LOG_DIR}/jockey_trainer_stats.log"
    else
      echo "(ログなし)"
    fi
    ;;

  *)
    echo "Usage: $0 {show|install|remove|status}"
    exit 1
    ;;
esac
