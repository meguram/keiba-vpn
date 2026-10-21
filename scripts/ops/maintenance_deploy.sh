#!/usr/bin/env bash
# メンテナンス時間帯（毎日 JST 03:00–04:00）の自動反映ランナーの起動ラッパー。ホストの root cron から JST 03:05（= 18:05 UTC）に呼ぶ。
# 実体は src/scripts/ops/maintenance_deploy.py。ステージ済み（data/local/deploy/pending/<id>/）の変更だけを適用する。
# 窓の外では何もしない。二重起動は flock で防ぐ。ログ: logs/maintenance_deploy.log、監査: data/local/meta/deploy_audit.jsonl
# このファイル自身は自動反映の対象外（人手で更新する）。
# Usage: bash scripts/ops/maintenance_deploy.sh [PROJECT_DIR] [-- 追加の引数（--dry-run / --force-window / --list）]
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
[ "$#" -gt 0 ] && shift
[ "${1:-}" = "--" ] && shift
cd "$PROJECT_DIR" || exit 1
mkdir -p logs data/local/deploy/pending
# Slack 通知（KEIBA_SLACK_NOTIFY=1 と SLACK_WEBHOOK_URL）を使うため .env を読む（値は表示しない）
if [ -f .env ]; then set -a; . ./.env 2>/dev/null; set +a; fi
export TZ="${TZ:-Asia/Tokyo}" KEIBA_ENV="${KEIBA_ENV:-stg}" PYTHONPATH=.
PY="${PYTHON:-/opt/conda/bin/python3}"; [ -x "$PY" ] || PY="$(command -v python3)"
exec 9>"/tmp/keiba_maintenance_deploy.sh.lock"
if ! flock -n 9; then echo "[$(date -Is)] すでに実行中。スキップ" >>logs/maintenance_deploy.log; exit 0; fi
echo "===== $(date -Is) START =====" >>logs/maintenance_deploy.log
"$PY" -m src.scripts.ops.maintenance_deploy "$@" >>logs/maintenance_deploy.log 2>&1
rc=$?
echo "===== $(date -Is) END rc=${rc} =====" >>logs/maintenance_deploy.log
exit "$rc"
