#!/usr/bin/env bash
# 取得を VPS に渡したあと、ホストに残す「後処理だけ」のジョブの cron を登録する（冪等。タグ KEIBA-HOST-POSTPROCESS の行だけ入れ替える）。
# ホストで root として実行。docs/operations/vps_only_fetch.md の段 4。
# 各ジョブは、対応する取得の担当が VPS に移ったときだけ動く（src/ops/host_postprocess.py）。移っていなければ何もしない（費用 0）。
# 時刻は cron のシステム時刻（UTC）。JST: 前夜の計算 19:10 と 19:40（取得の遅れを吸収。GCP Jobs と同じ 2 回）、馬場速度 07:20、
# 予測の成績 土日月の 21:50（開催日の夜に 1 回。当日分は速報のランナーが日の終わりに集計する）。各ジョブにロックと上限時間（host_postprocess.py）。
# 費用（最悪・月。ホストは GCP の外なので、VPS が保存したデータを読む分が転送料）: docs/operations/vps_only_fetch.md の段 4 の見積もり（合計 約 20 円）。
# Usage: bash scripts/cron/install_host_postprocess_cron.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-HOST-POSTPROCESS"
PY="${PY:-/opt/conda/bin/python3}"
W="cd \"${PROJECT_DIR}\" && TZ=Asia/Tokyo bash \"${PROJECT_DIR}/scripts/cron/cron_run.sh\""
L=">> \"${PROJECT_DIR}/logs/host_postprocess.log\" 2>&1 # ${TAG}"
E="env KEIBA_ENV=stg PYTHONPATH=. \"${PY}\" -m src.ops.host_postprocess"
LINES="10,40 10 * * * ${W} host-eve-precompute -- ${E} eve-precompute ${L}
20 22 * * * ${W} host-track-speed-daily -- ${E} track-speed-daily ${L}
50 12 * * 0,1,6 ${W} host-prediction-results -- ${E} prediction-results-refresh ${L}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data/local/meta/cron_backups"
crontab -l 2>/dev/null > "${PROJECT_DIR}/data/local/meta/cron_backups/crontab_before_host_postprocess_$(date +%Y%m%d_%H%M%S).txt" || true
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -v "# ${TAG}" > "$TMP" || true
printf '%s\n' "$LINES" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "登録しました（${TAG}）:"; printf '%s\n' "$LINES"
