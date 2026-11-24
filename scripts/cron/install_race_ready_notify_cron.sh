#!/usr/bin/env bash
# レースごとの「予測・推奨が画面に出た」Slack 通知の cron を入れる（冪等。タグ KEIBA-RACE-READY-NOTIFY の行だけを入れ替える）。ホストで root として実行する。
#   - cron: 毎分、JST 07:00〜17:59（root の crontab は UTC なので 22-23,0-8 時）。非開催日はすぐ終わる。
#   - 目印 data/local/meta/race_ready_notify.on を置く（送り始める）。--no-mark は cron だけ入れて目印は置かない（送らずにログだけ）。
#   - --remove で行と目印を消す。登録前に crontab -l を data/local/meta/cron_backups/ へ退避する。
# 追跡されるファイルは書き換えない（共有の作業ツリーの毎時の pull を止めない）。手順: docs/operations/slack_race_ready_plan_20261119.md
set -eu
[ "$(id -u)" = 0 ] || { echo "root で実行してください（push・ランナーと同じ root の crontab に入れる）" >&2; exit 1; }
PROJECT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
TAG="KEIBA-RACE-READY-NOTIFY"
CRON_TIME="* 22-23,0-8 * * *"
LINE="${CRON_TIME} bash ${PROJECT_DIR}/scripts/cron/race_ready_notify.sh ${PROJECT_DIR} >> ${PROJECT_DIR}/logs/race_ready_notify.cron.log 2>&1 # ${TAG}"
META="${PROJECT_DIR}/data/local/meta"
MARK="${META}/race_ready_notify.on"
mkdir -p "${PROJECT_DIR}/logs" "${META}/cron_backups"
BK="${META}/cron_backups/crontab_before_race_ready_notify_$(date +%Y%m%d_%H%M%S).txt"
crontab -l 2>/dev/null > "$BK" || true
echo "crontab を退避しました: $BK"
TMP="$(mktemp)"
grep -v "# ${TAG}" "$BK" > "$TMP" || true
case "${1:-}" in
  --remove) rm -f "$MARK" ;;
  --no-mark) echo "$LINE" >> "$TMP" ;;
  "") echo "$LINE" >> "$TMP"; date '+%Y-%m-%dT%H:%M:%S' > "$MARK" ;;
  *) echo "使い方: $0 [--no-mark|--remove]" >&2; rm -f "$TMP"; exit 2 ;;
esac
crontab "$TMP"
rm -f "$TMP"
case "${1:-}" in
  --remove) echo "消しました（${TAG} の行と目印）" ;;
  --no-mark) echo "登録しました（目印なし＝送らずにログだけ）:"; echo "$LINE" ;;
  *) echo "登録しました:"; echo "$LINE"; echo "目印: $MARK" ;;
esac
