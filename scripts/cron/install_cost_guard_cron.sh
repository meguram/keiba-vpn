#!/usr/bin/env bash
# 費用の歯止め（src.ops.cost_guard）の cron を登録する（冪等。タグ KEIBA-COST-GUARD の行だけを入れ替える）。ホストで root として実行。
# 今月の GCP 費用の累計か月末の見込みが上限（config/cost_assumptions.json の budget_jpy_per_month）の 80% 以上なら、
# 並走中の GCP Jobs の Cloud Scheduler（keiba-g-*。費用の日報 keiba-g-0730-shadow は残す）を一時停止して Slack に知らせる。自動では再開しない。
# 時刻は cron のシステム時刻（UTC）。JST 10:17 と 22:17（01:17 / 13:17 UTC）。課金エクスポートの反映は約 1 日遅れる。
# KEIBA_ENV=stg: 記録（ops/cost_guard/）は .env.stg の GCS_BUCKET に書く（ホストの他の記録と同じ）。
# 費用: BigQuery の照会は 1 回数 MB で無料枠の範囲。GCS は記録を 1 日 2 回書くだけ（Class A 2 回）。
# Usage: bash scripts/cron/install_cost_guard_cron.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-COST-GUARD"
PY="${PY:-/opt/conda/bin/python3}"
LINES="17 1,13 * * * cd \"${PROJECT_DIR}\" && TZ=Asia/Tokyo bash \"${PROJECT_DIR}/scripts/cron/cron_run.sh\" cost-guard -- env KEIBA_ENV=stg PYTHONPATH=. \"${PY}\" -m src.ops.cost_guard --apply >> \"${PROJECT_DIR}/logs/cost_guard.log\" 2>&1 # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data/local/meta/cron_backups"
crontab -l 2>/dev/null > "${PROJECT_DIR}/data/local/meta/cron_backups/crontab_before_cost_guard_$(date +%Y%m%d_%H%M%S).txt" || true
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -v "# ${TAG}" > "$TMP" || true
printf '%s\n' "$LINES" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "登録しました（${TAG}）:"; printf '%s\n' "$LINES"
