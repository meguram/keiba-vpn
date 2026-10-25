#!/usr/bin/env bash
# 優先処理（2020-2025 年の取り直し: data/local/meta/rescrape/run_priority_rescrape.sh）が動いている間、
# netkeiba のログインを使う定期ジョブを見送る。同じアカウントで別セッションがログインすると、取り直し側のセッションが無効になるため。
# 判定は『bash <パス>/run_priority_rescrape.sh』で始まるプロセスだけ（ssh や grep のコマンドラインに名前が含まれるだけでは一致させない）。
# 優先処理が終われば（プロセスが無くなれば）自動で通常どおり動く。止めたいときは YIELD_DISABLED ファイルを置く。
#
# Usage: yield_to_rescrape.sh <label> -- <command...>
set -u
LABEL="${1:?label}"; shift
[ "${1:-}" = "--" ] && shift
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DISABLED="${PROJECT_DIR}/data/local/meta/rescrape/YIELD_DISABLED"
if [ ! -e "$DISABLED" ] && pgrep -f '^(/bin/|/usr/bin/)?bash( -[a-zA-Z]+)* [^ ]*run_priority_rescrape\.sh' >/dev/null 2>&1; then
  mkdir -p "${PROJECT_DIR}/logs"
  MSG="[$(TZ=Asia/Tokyo date -Is)] SKIP ${LABEL}: 優先処理(取り直し)が動いているため見送り"
  echo "$MSG" >>"${PROJECT_DIR}/logs/yield_skips.log"
  echo "$MSG"       # 標準出力にも出す（cron_run.sh の記録に残り、監視ポータルで『見送り』と判別できる）
  exit 0
fi
exec "$@"
