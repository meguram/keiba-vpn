#!/usr/bin/env bash
# root の crontab に重なって残った setup_all_cron.sh の見出しのブロック（タグの無いコメント行と PATH=）を、最初の 1 回分だけ残して消す。
# 2026-11-13 時点で 6 回分重なっていた（remove_all がタグの無い行を消さなかったため。setup_all_cron.sh は直した）。
# setup_all_cron.sh install は使わない（生成する行が、その後に手で整理した crontab より古く、廃止したジョブを戻すため）。
# 守り: root でなければ・crontab -l が失敗したら・有効な行が 20 未満なら中止。退避する。有効な行（コメントと空行以外）の数が変わったら中止。
# Usage: sudo bash scripts/cron/dedupe_setup_all_headers.sh [--dry-run]
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
if [ "${KEIBA_CRON_ALLOW_NONROOT:-0}" != 1 ] && [ "$(id -u)" != 0 ]; then echo "root で実行してください。中止" >&2; exit 1; fi
NOW="$(mktemp)"; NEW="$(mktemp)"; ERR="$(mktemp)"; trap 'rm -f "$NOW" "$NEW" "$ERR"' EXIT
crontab -l >"$NOW" 2>"$ERR" || { echo "crontab -l に失敗したので中止します: $(cat "$ERR")" >&2; exit 1; }
before=$(grep -vcE '^[[:space:]]*(#|$)' "$NOW" || true)
[ "$before" -ge "${KEIBA_CRON_MIN_LINES:-20}" ] || { echo "有効な行が少なすぎる（${before}）。中止" >&2; exit 1; }
# 見出しのブロックの行（setup_all_cron.sh の generate_cron のうち、タグの無い行）
HDR="$(bash "${PROJECT_DIR}/scripts/cron/setup_all_cron.sh" show 2>/dev/null | grep -vE '# KEIBA|^[[:space:]]*$' | sort -u || true)"
[ -n "$HDR" ] || { echo "見出しの行を取れなかったので中止" >&2; exit 1; }
awk -v hdr="$HDR" 'BEGIN { n = split(hdr, a, "\n"); for (i = 1; i <= n; i++) h[a[i]] = 1 }
  { if (($0 in h) && seen[$0]++) next; print }' "$NOW" | cat -s >"$NEW"
after=$(grep -vcE '^[[:space:]]*(#|$)' "$NEW" || true)
if [ "$(grep -c '^PATH=' "$NOW" || true)" -gt 0 ]; then before=$((before - $(grep -c '^PATH=' "$NOW") + 1)); fi   # PATH= は有効な行として数えられ、重複の分だけ減る
[ "$after" -eq "$before" ] || { echo "有効な行の数が合いません（前 ${before}・後 ${after}）。中止" >&2; exit 1; }
diff "$NOW" "$NEW" | head -80 || true
if [ "$DRY" = 1 ]; then echo "（--dry-run なので変えていません）"; exit 0; fi
BK="${KEIBA_CRON_BACKUP_DIR:-${PROJECT_DIR}/data/local/meta/cron_backups}"; mkdir -p "$BK"
cp "$NOW" "${BK}/crontab_before_dedupe_headers_$(date +%Y%m%d_%H%M%S).txt"
crontab "$NEW"
echo "見出しの重なりを消しました（有効な行 ${after}）"
