#!/usr/bin/env bash
# 影のランナーの材料の予約（candidate_rb_live.sh）を cron に入れる（冪等。タグ KEIBA-CANDIDATE-RB の行だけを入れ替える）。ホストで root として実行する。
#   - 09:30 UTC = 18:30 JST に eve（翌日が開催日なら前段と RB）
#   - 22:05〜22:40 UTC = 07:05〜07:40 JST に 5 分おきに morning（前の開催日の結果が表にそろったら RB を作り直す）
# root で動くので、rb_live/ の下にできる state_*・prep_*・ログは root の持ち物になる。手で流すときも sudo で流す。
# --remove で行を消す。登録前に crontab -l を data/local/meta/cron_backups/ へ退避する。手順: docs/operations/weekend_launch_20261121.md §4
set -eu
PROJECT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
case "$PROJECT_DIR" in
  */.claude/worktrees/*) echo "共有の作業ツリー（/home/jovyan/work/keiba-vpn）の scripts/cron から流す（worktree から入れると cron が worktree を指す）" >&2; exit 1 ;;
esac
TAG="KEIBA-CANDIDATE-RB"
RUN="bash ${PROJECT_DIR}/scripts/cron/candidate_rb_live.sh"
LOG=">> ${PROJECT_DIR}/logs/candidate_rb_live.cron.log 2>&1"
L1="30 9 * * * ${RUN} eve ${PROJECT_DIR} ${LOG} # ${TAG}"
L2="5-40/5 22 * * * ${RUN} morning ${PROJECT_DIR} ${LOG} # ${TAG}"
META="${PROJECT_DIR}/data/local/meta"
mkdir -p "${PROJECT_DIR}/logs" "${META}/cron_backups"
BK="${META}/cron_backups/crontab_before_candidate_rb_$(date +%Y%m%d_%H%M%S).txt"
crontab -l 2>/dev/null > "$BK" || true
echo "crontab を退避しました: $BK"
TMP="$(mktemp)"
grep -v "# ${TAG}" "$BK" > "$TMP" || true
if [ "${1:-}" != "--remove" ]; then
  echo "$L1" >> "$TMP"
  echo "$L2" >> "$TMP"
fi
crontab "$TMP"
rm -f "$TMP"
if [ "${1:-}" = "--remove" ]; then echo "消しました（${TAG} の行）"; else echo "登録しました:"; echo "$L1"; echo "$L2"; fi
