#!/usr/bin/env bash
# Sporthorse 夜間バッチを crontab に登録（冪等。タグ KEIBA-SPORTHORSE-BIRTHS の行だけ入れ替え）。
# Usage: bash scripts/cron/install_sporthorse_births_cron.sh [PROJECT_DIR]
set -eu
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-SPORTHORSE-BIRTHS"
# 毎日 22:00 JST = 13:00 UTC
LINE="0 13 * * * cd ${PROJECT_DIR} && TZ=Asia/Tokyo bash ${PROJECT_DIR}/scripts/cron/cron_run.sh sporthorse-births-nightly -- bash ${PROJECT_DIR}/scripts/cron/sporthorse_births_nightly.sh ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/sporthorse_births_nightly.log 2>&1 # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs"
NIGHTLY="${PROJECT_DIR}/scripts/cron/sporthorse_births_nightly.sh"
if [ ! -x "$NIGHTLY" ]; then
  chmod +x "$NIGHTLY" 2>/dev/null || sudo chmod +x "$NIGHTLY" 2>/dev/null || true
fi
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -v "# ${TAG}" > "$TMP" || true
echo "$LINE" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "登録しました:"
echo "  $LINE"
echo ""
echo "事前確認（学習PC上）:"
echo "  bash ${PROJECT_DIR}/scripts/cron/sporthorse_births_nightly.sh ${PROJECT_DIR}  # セッション無しなら SKIP"
echo "  ${PYTHON:-python3} -m src.research.pedigree.sporthorse_births status --ancestors /tmp/ancestor_birthdays.csv --zone ${PROJECT_DIR}/data/meta/dormello/zone_review.csv --nk /tmp/nk_birth_results.jsonl --dir ${PROJECT_DIR}/data/meta/dormello"
