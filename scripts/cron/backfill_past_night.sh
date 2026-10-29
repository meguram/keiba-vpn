#!/usr/bin/env bash
# 過去分（2016〜2019 年）の補填を、時間帯・曜日で分けず一定ペース（1 秒おき。他の取得とは並走する）で進める（毎時起動、実行中なら何もしない）。
# 通信エラーのみ（HTTP 403/429 なし）の HALT は次回クロン時に自動解除し再開する。HTTP 制限が検知された HALT は人手確認まで再開しない。
# 取得の強さの制限（1 回の件数の上限・段階的な引き上げ・悪化の検知による中断・日の総件数の上限・他の取得が動いていれば待つ）は
# src/scraper/backfill_guard.py と backfill.py の --phase past が行う。仕様・運用: docs/operations/backfill_2016.md
# Usage: bash scripts/cron/backfill_past_night.sh [PROJECT_DIR] [追加の引数...]
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
[ $# -gt 0 ] && shift
cd "$PROJECT_DIR" || exit 1
mkdir -p logs
LOG="logs/backfill_past.log"
PY="${PYTHON:-/opt/conda/bin/python3}"
HALT="$PROJECT_DIR/data/local/meta/backfill_past_halt.json"

# HALT ファイルが「通信エラーのみ（bad=0 ＝ HTTP 403/429/5xx なし）」であれば自動解除して再開する。
# bot 検知（bad>0）の場合は人手確認が必要なため自動解除しない。
if [ -f "$HALT" ]; then
  bad=$("$PY" -c "import json; d=json.load(open('$HALT')); print(d.get('bad',1))" 2>/dev/null || echo "1")
  age_min=$(( ($(date +%s) - $(date -r "$HALT" +%s 2>/dev/null || echo 0)) / 60 ))
  if [ "$bad" = "0" ]; then
    echo "[$(date -Is)] HALT 自動解除: 通信エラーのみ（bad=0）・${age_min}分経過。次回クロンで再開します" >>"$LOG"
    rm -f "$HALT"
  else
    echo "[$(date -Is)] HALT 継続: HTTP エラーあり（bad=${bad}）・人手確認が必要です（rm $HALT）" >>"$LOG"
  fi
fi

exec 9>"/tmp/keiba_backfill_past.lock"
if ! flock -n 9; then echo "[$(date -Is)] すでに実行中。スキップ" >>"$LOG"; exit 0; fi
echo "===== $(date -Is) START =====" >>"$LOG"
# BACKFILL_PAST_MAX_TRANSPORT_ERRORS=3: 通信エラー 3 件まで許容してから中断（デフォルト 1 から緩和）
# 固まり検知（src/scraper/stall_watchdog.py）が強制終了した場合（rc=75）は、次の毎時起動を待たず 60 秒置いて最大 5 回まで再起動する。
RC=0
for attempt in 1 2 3 4 5; do
  # 1 秒おき: 待ちを 1 秒に固定し、14 件・60 件ごとの追加の休み（バースト・セッションのクールダウン）は 0 秒にする（2026-10-29 ユーザー指示）。
  # 他の取得に譲らない（BACKFILL_PAST_YIELD=0）。同時の取得はプロセス横断スロット（NETKEIBA_GLOBAL_MAX_SLOTS、既定 2）で抑える
  NETKEIBA_THROTTLE_MIN="${NETKEIBA_THROTTLE_MIN:-1.0}" NETKEIBA_THROTTLE_MAX="${NETKEIBA_THROTTLE_MAX:-1.0}" \
    NETKEIBA_BURST_COOLDOWN_MIN="${NETKEIBA_BURST_COOLDOWN_MIN:-0}" NETKEIBA_BURST_COOLDOWN_MAX="${NETKEIBA_BURST_COOLDOWN_MAX:-0}" \
    NETKEIBA_SESSION_COOLDOWN_MIN="${NETKEIBA_SESSION_COOLDOWN_MIN:-0}" NETKEIBA_SESSION_COOLDOWN_MAX="${NETKEIBA_SESSION_COOLDOWN_MAX:-0}" \
    BACKFILL_PAST_YIELD="${BACKFILL_PAST_YIELD:-0}" \
    KEIBA_ENV="${KEIBA_ENV:-stg}" BACKFILL_PAST_MAX_TRANSPORT_ERRORS="${BACKFILL_PAST_MAX_TRANSPORT_ERRORS:-3}" \
    PYTHONPATH=. nice -n 10 "$PY" -m src.scraper.backfill --phase past "$@" >>"$LOG" 2>&1
  RC=$?
  echo "===== $(date -Is) END rc=${RC} (試行 ${attempt}) =====" >>"$LOG"
  [ "$RC" -eq 75 ] || break
  sleep 60
done
exit "$RC"
