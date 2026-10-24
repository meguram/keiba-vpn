#!/usr/bin/env bash
# ローカル PostgreSQL（Debian/Ubuntu のクラスタ。ホストでは 14/main）が止まっていれば起動する。冪等。
#
# cron から呼ぶ想定（コンテナ起動時と数分おき。root の crontab）:
#   @reboot sleep 20 && /path/to/keiba-vpn/scripts/server/ensure_postgres.sh
#   */5 * * * * /path/to/keiba-vpn/scripts/server/ensure_postgres.sh
#
# 背景: ホストはコンテナで systemd が無く、再起動のたびに PostgreSQL が落ちたままになる。
# 2026-10-14〜15 に夜間の GCS → PG 同期が接続拒否で失敗していた原因（クラスタ自体は残っており、起動するだけで復旧した）。
# ログ: logs/ensure_postgres.log

set -uo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
LOG="$ROOT/logs/ensure_postgres.log"
mkdir -p "$ROOT/logs"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo -n"

command -v pg_lsclusters >/dev/null 2>&1 || exit 0   # PostgreSQL が入っていない環境（dev など）では何もしない

started=0
while read -r ver name _port status _rest; do
  [ "$ver" = "Ver" ] && continue
  [ -z "$ver" ] && continue
  if [ "$status" = "down" ]; then
    out="$($SUDO pg_ctlcluster "$ver" "$name" start 2>&1)"
    rc=$?
    echo "$(date -Is) start $ver/$name rc=$rc ${out}" >>"$LOG"
    started=1
  fi
done < <($SUDO pg_lsclusters 2>/dev/null)
exit 0
