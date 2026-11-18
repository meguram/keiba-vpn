#!/usr/bin/env bash
# 候補の影のランナーを有効にする（冪等。タグ KEIBA-CANDIDATE-RUNNER の行だけを入れ替える）。ホストで root として実行する。
#   - 目印 data/local/meta/candidate_runner.on と data/local/meta/odds_t60_copy.on を置く（追跡されるファイルは書き換えない。毎時の pull を止めない）
#   - cron: 22:45 UTC = 07:45 JST に起動（開催の無い日は発走時刻表を見てすぐ終わる）。落ちたときのために 08:00〜16:45 JST に 15 分おきに
#     起動し直す（flock で二重に動かない。再起動しても判定済み・発走済みのレースは書き直さない）。
# --remove で行と目印を消す（戻すとき）。登録前に crontab -l を data/local/meta/cron_backups/ へ退避する。手順: docs/operations/weekend_launch_20261121.md
set -eu
PROJECT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
TAG="KEIBA-CANDIDATE-RUNNER"
RUN="KEIBA_CANDIDATE_RUNNER=1 bash ${PROJECT_DIR}/scripts/cron/candidate_runner.sh ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/candidate_runner.cron.log 2>&1"
L1="45 22 * * * ${RUN} # ${TAG}"
L2="*/15 23,0-7 * * * ${RUN} # ${TAG}"
META="${PROJECT_DIR}/data/local/meta"
mkdir -p "${PROJECT_DIR}/logs" "${META}/cron_backups"
BK="${META}/cron_backups/crontab_before_candidate_runner_$(date +%Y%m%d_%H%M%S).txt"
crontab -l 2>/dev/null > "$BK" || true
echo "crontab を退避しました: $BK"
TMP="$(mktemp)"
grep -v "# ${TAG}" "$BK" > "$TMP" || true
if [ "${1:-}" = "--remove" ]; then
  rm -f "${META}/candidate_runner.on" "${META}/odds_t60_copy.on"
else
  echo "$L1" >> "$TMP"
  echo "$L2" >> "$TMP"
  date '+%Y-%m-%dT%H:%M:%S' > "${META}/candidate_runner.on"
  date '+%Y-%m-%dT%H:%M:%S' > "${META}/odds_t60_copy.on"
fi
crontab "$TMP"
rm -f "$TMP"
if [ "${1:-}" = "--remove" ]; then echo "消しました（${TAG} の行と目印）"; else echo "登録しました:"; echo "$L1"; echo "$L2"; echo "目印: ${META}/candidate_runner.on・odds_t60_copy.on"; fi
