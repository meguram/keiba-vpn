#!/bin/bash
# =============================================================================
# Backfill cron ジョブ設定スクリプト
#
# 過去データの自動取得を cron で設定する。
# 年度ごとに並列実行し、各ジョブは 1回あたり最大5日分を処理する。
#
# 使い方:
#   bash scripts/setup_backfill_cron.sh          # cron 設定を表示
#   bash scripts/setup_backfill_cron.sh install   # cron に登録
#   bash scripts/setup_backfill_cron.sh remove    # cron から削除
#   bash scripts/setup_backfill_cron.sh status    # 進捗確認
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="python3"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "$LOG_DIR"

CRON_TAG="# KEIBA_BACKFILL"

generate_cron() {
    cat <<CRON_ENTRIES
# =========================================================
# ML-AutoPilot Keiba: 過去データ自動取得 (Backfill)
# 4並列 × 年度分割で効率的にデータ収集
# =========================================================

# --- Phase fast: レース結果 + 出馬表 (毎日 AM 1:00, 2:00, 3:00, 4:00) ---
0  1 * * * cd ${PROJECT_DIR} && ${PYTHON} -m scraper.backfill --year 2025 --phase fast --max-dates 5 >> ${LOG_DIR}/backfill_2025.log 2>&1 ${CRON_TAG}
0  2 * * * cd ${PROJECT_DIR} && ${PYTHON} -m scraper.backfill --year 2024 --phase fast --max-dates 5 >> ${LOG_DIR}/backfill_2024.log 2>&1 ${CRON_TAG}
0  3 * * * cd ${PROJECT_DIR} && ${PYTHON} -m scraper.backfill --year 2023 --phase fast --max-dates 5 >> ${LOG_DIR}/backfill_2023.log 2>&1 ${CRON_TAG}
0  4 * * * cd ${PROJECT_DIR} && ${PYTHON} -m scraper.backfill --year 2022 --phase fast --max-dates 5 >> ${LOG_DIR}/backfill_2022.log 2>&1 ${CRON_TAG}

# --- Phase horse: 馬情報一括取得 (毎日 AM 6:00) ---
0  6 * * * cd ${PROJECT_DIR} && ${PYTHON} -m scraper.backfill --phase horse >> ${LOG_DIR}/backfill_horse.log 2>&1 ${CRON_TAG}

# --- Phase full: 補助データ (毎日 AM 8:00, 9:00) ---
0  8 * * * cd ${PROJECT_DIR} && ${PYTHON} -m scraper.backfill --year 2025 --phase full --max-dates 3 >> ${LOG_DIR}/backfill_full_2025.log 2>&1 ${CRON_TAG}
0  9 * * * cd ${PROJECT_DIR} && ${PYTHON} -m scraper.backfill --year 2024 --phase full --max-dates 3 >> ${LOG_DIR}/backfill_full_2024.log 2>&1 ${CRON_TAG}

# --- 2020-2021 (古い年度は週2回で十分) ---
0  2 * * 1,4 cd ${PROJECT_DIR} && ${PYTHON} -m scraper.backfill --year 2021 --phase fast --max-dates 5 >> ${LOG_DIR}/backfill_2021.log 2>&1 ${CRON_TAG}
0  3 * * 2,5 cd ${PROJECT_DIR} && ${PYTHON} -m scraper.backfill --year 2020 --phase fast --max-dates 5 >> ${LOG_DIR}/backfill_2020.log 2>&1 ${CRON_TAG}

# --- ログローテーション (毎週月曜 AM 0:00) ---
0  0 * * 1 find ${LOG_DIR} -name "backfill_*.log" -size +10M -exec truncate -s 0 {} \; ${CRON_TAG}

CRON_ENTRIES
}

case "${1:-show}" in
    show)
        echo "以下の cron エントリが生成されます:"
        echo ""
        generate_cron
        echo ""
        echo "登録するには: bash $0 install"
        ;;

    install)
        existing=$(crontab -l 2>/dev/null || true)
        cleaned=$(echo "$existing" | grep -v "$CRON_TAG" || true)

        new_entries=$(generate_cron)
        echo "$cleaned" > /tmp/keiba_cron_tmp
        echo "" >> /tmp/keiba_cron_tmp
        echo "$new_entries" >> /tmp/keiba_cron_tmp

        crontab /tmp/keiba_cron_tmp
        rm /tmp/keiba_cron_tmp

        echo "✅ cron ジョブを登録しました"
        echo ""
        echo "確認: crontab -l | grep KEIBA_BACKFILL"
        echo "ログ: ls -la ${LOG_DIR}/backfill_*.log"
        ;;

    remove)
        existing=$(crontab -l 2>/dev/null || true)
        cleaned=$(echo "$existing" | grep -v "$CRON_TAG" || true)
        echo "$cleaned" | crontab -
        echo "✅ KEIBA_BACKFILL cron ジョブを削除しました"
        ;;

    status)
        cd "$PROJECT_DIR"
        $PYTHON -m scraper.backfill --status
        echo ""
        echo "=== 直近のログ ==="
        for f in ${LOG_DIR}/backfill_*.log; do
            if [ -f "$f" ]; then
                lines=$(wc -l < "$f")
                last=$(tail -1 "$f" 2>/dev/null | head -c 100)
                echo "  $(basename "$f"): ${lines} 行 | ${last}"
            fi
        done
        ;;

    *)
        echo "Usage: $0 {show|install|remove|status}"
        exit 1
        ;;
esac
