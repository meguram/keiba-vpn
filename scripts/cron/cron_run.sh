#!/usr/bin/env bash
# cron ジョブの実行を記録する共通ラッパー。監視ポータル(src/monitor/cron_status.py)が読む。
#
# Usage: cron_run.sh <job_id> -- <command...>
#   例: bash scripts/cron/cron_run.sh sync_pg_from_gcs -- bash scripts/cron/sync_pg_from_gcs.sh
#
# 元のコマンドの出力はそのまま標準出力へ流す（crontab の >> リダイレクトにそのまま入る）。終了コードも元のコマンドのまま返す。
# 記録は data/local/meta/cron_runs/（KEIBA_CRON_RUNS_DIR で変更可）に置く:
#   <job_id>.running    実行中マーカー「pid 開始epoch」。終了時に消える。クラッシュで残っても pid の生死で判定できる
#   <job_id>.tsv        完了ごとに 1 行「開始epoch 終了epoch 終了コード pid」。直近 200 行に切り詰める
#   <job_id>.tail.txt   直近の実行の出力の末尾 80 行（上書き）
#   <job_id>.fail.txt   直近の『失敗』した実行の出力の末尾 80 行（先頭に時刻と終了コード。成功しても消さない）
# 記録に失敗しても（ディレクトリが書けない等）、元のコマンドは必ず実行する。シェルだけで書く（*/2 分おきのジョブがあるため）。
set -u
JOB="${1:?job_id}"; shift
[ "${1:-}" = "--" ] && shift
[ "$#" -gt 0 ] || { echo "usage: cron_run.sh <job_id> -- <command...>" >&2; exit 2; }
JOB="$(printf '%s' "$JOB" | tr -c 'A-Za-z0-9._-' '_')"
PROJECT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
# GCP / VPS が引き継いだジョブ（config/job_owner.json）は、ホストでは実行しない。記録は成功として残す（監視が『止まった』と誤判定しない）。
# ファイルが無い・python が無い・判定に失敗したときは、これまでどおり実行する。
if [ -f "${PROJECT_DIR}/config/job_owner.json" ] && command -v python3 >/dev/null 2>&1; then
  PYTHONPATH="$PROJECT_DIR" python3 -m src.ops.job_owner check "$JOB" >/dev/null 2>&1
  if [ "$?" = 10 ]; then
    set -- bash -c "echo \"[\$(date -Is)] ${JOB} は GCP / VPS の担当（config/job_owner.json）。このホストでは実行しません\"; exit 0"
  fi
fi
DIR="${KEIBA_CRON_RUNS_DIR:-${PROJECT_DIR}/data/local/meta/cron_runs}"
KEEP_LINES=200
TAIL_LINES=80

REC=1
mkdir -p "$DIR" 2>/dev/null && [ -w "$DIR" ] || REC=0
START="$(date +%s)"
TAILF="" ; TAILOK=""
if [ "$REC" = 1 ]; then
  TAILF="$(mktemp "${DIR}/.${JOB}.XXXXXX" 2>/dev/null)" || REC=0
fi
if [ "$REC" = 1 ]; then
  TAILOK="${TAILF}.ok"
  printf '%s %s\n' "$$" "$START" >"${DIR}/${JOB}.running" 2>/dev/null || true
  CHILD=""
  on_signal() { [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null; }
  trap on_signal TERM INT HUP
  # 出力は標準出力へそのまま流しつつ、末尾 80 行だけを別に取る（長時間のランナーでも肥大しない）
  "$@" > >(tee >(tail -n "$TAIL_LINES" >"$TAILF" 2>/dev/null; : >"$TAILOK" 2>/dev/null)) 2>&1 &
  CHILD=$!
  wait "$CHILD"; EC=$?
  while kill -0 "$CHILD" 2>/dev/null; do wait "$CHILD"; EC=$?; done      # シグナルで wait が先に戻っても、子の本当の終了コードを待つ
  trap - TERM INT HUP
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do [ -e "$TAILOK" ] && break; sleep 0.1; done
  END="$(date +%s)"
  {
    printf '%s\t%s\t%s\t%s\n' "$START" "$END" "$EC" "$$" >>"${DIR}/${JOB}.tsv"
    if [ "$(wc -l <"${DIR}/${JOB}.tsv")" -gt $((KEEP_LINES + 60)) ]; then
      tail -n "$KEEP_LINES" "${DIR}/${JOB}.tsv" >"${DIR}/.${JOB}.tsv.$$" && mv -f "${DIR}/.${JOB}.tsv.$$" "${DIR}/${JOB}.tsv"
    fi
    if [ -s "$TAILF" ] || [ -e "$TAILOK" ]; then
      if [ "$EC" -ne 0 ]; then
        { printf '# %s exit=%s\n' "$(date -Is 2>/dev/null || date)" "$EC"; cat "$TAILF"; } >"${DIR}/.${JOB}.fail.$$" && chmod 644 "${DIR}/.${JOB}.fail.$$" 2>/dev/null; mv -f "${DIR}/.${JOB}.fail.$$" "${DIR}/${JOB}.fail.txt"
      fi
      chmod 644 "$TAILF" 2>/dev/null
      mv -f "$TAILF" "${DIR}/${JOB}.tail.txt"
    fi
    rm -f "${DIR}/${JOB}.running" "$TAILOK" "$TAILF"
  } 2>/dev/null || true
  exit "$EC"
fi
exec "$@"
