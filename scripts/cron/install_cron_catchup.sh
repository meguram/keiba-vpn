#!/usr/bin/env bash
# PC（cron）が止まっていて予定時刻を逃した毎日のジョブを、起動後に 1 回だけ追いつき実行する cron を登録／削除する（冪等。タグ KEIBA-CRON-CATCHUP）。
# 15 分おきに src.ops.cron_catchup run を呼ぶ（読むのはローカルのファイルだけ。対象は config/cron_catchup.json）。ホストで root として実行。
# 登録はユーザーの承認後に行うこと（docs/operations/cron_catchup.md）。
# Usage: bash scripts/cron/install_cron_catchup.sh [PROJECT_DIR] | --remove [PROJECT_DIR]
set -eu
MODE=install
if [ "${1:-}" = "--remove" ]; then MODE=remove; shift; fi
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
TAG="KEIBA-CRON-CATCHUP"
LINE="*/15 * * * * cd ${PROJECT_DIR} && bash ${PROJECT_DIR}/scripts/cron/cron_run.sh cron-catchup -- env PYTHONPATH=. /opt/conda/bin/python3 -m src.ops.cron_catchup run >> ${PROJECT_DIR}/logs/cron_catchup.log 2>&1 # ${TAG}"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data/local/meta/cron_backups"
# crontab -l が「crontab が無い」以外で失敗したら止める（空の内容で crontab 全体を置き換えないように）
if CUR="$(crontab -l 2>&1)"; then :; elif printf '%s' "$CUR" | grep -qi "no crontab"; then CUR=""; else
  echo "crontab -l に失敗したので中止します: $CUR" >&2; exit 1
fi
printf '%s\n' "$CUR" > "${PROJECT_DIR}/data/local/meta/cron_backups/crontab_before_cron_catchup_$(date +%Y%m%d_%H%M%S).txt"
TMP="$(mktemp)"
printf '%s\n' "$CUR" | grep -v "# ${TAG}" > "$TMP" || true
if [ "$MODE" = install ]; then echo "$LINE" >> "$TMP"; fi
crontab "$TMP"
rm -f "$TMP"
if [ "$MODE" = install ]; then echo "登録しました: ${LINE}"; else echo "削除しました（タグ ${TAG}）"; fi
