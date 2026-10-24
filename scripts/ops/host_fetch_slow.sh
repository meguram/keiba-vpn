#!/usr/bin/env bash
# 二重アクセス(ホスト + VPS の影モード)の期間中、ホストの過去分補填(backfill-past)を控えめにする。root の crontab の該当 1 行だけを書き換える。
#   bash host_fetch_slow.sh status            # いまの設定を表示（何も変えない）
#   bash host_fetch_slow.sh apply             # 取得の間隔を 4.0〜5.0 秒にする（既定は 1.8〜2.2 秒。約 2 倍ゆっくり）
#   bash host_fetch_slow.sh revert            # 元に戻す
#   --file <crontab 形式のファイル>            # crontab を使わず、このファイルを書き換える（確認用）
# 対象は『# KEIBA-BACKFILL-PAST』の行の backfill_past_night.sh の起動だけ。開催日の取得(entries-sync / raceday-*)・他の補填には触れない。
# 変更前の crontab は data/local/meta/cron_backups/ に退避する。何度実行しても同じ結果。root で実行（crontab を書く）。
set -euo pipefail
ACTION="${1:-status}"; shift || true
FILE=""; MIN="${THROTTLE_MIN:-4.0}"; MAX="${THROTTLE_MAX:-5.0}"
while [ $# -gt 0 ]; do case "$1" in --file) FILE="${2:?}"; shift 2;; *) echo "不明な引数: $1" >&2; exit 2;; esac; done
TAG='# KEIBA-BACKFILL-PAST'
MARK="NETKEIBA_THROTTLE_MIN=${MIN} NETKEIBA_THROTTLE_MAX=${MAX} "
read_cron() { if [ -n "$FILE" ]; then cat "$FILE"; else crontab -l; fi; }
write_cron() { if [ -n "$FILE" ]; then cat > "$FILE"; else crontab -; fi; }
CUR="$(read_cron)"
LINE="$(printf '%s\n' "$CUR" | grep -F -- "$TAG" | grep -F 'backfill_past_night.sh' || true)"
[ -n "$LINE" ] || { echo "対象の行（${TAG}）が見つかりません" >&2; exit 3; }
[ "$(printf '%s\n' "$LINE" | wc -l)" = 1 ] || { echo "対象の行が複数あります。手で確認してください" >&2; exit 3; }
status() { if printf '%s' "$LINE" | grep -q 'NETKEIBA_THROTTLE_MIN='; then echo "控えめ: $(printf '%s' "$LINE" | grep -o 'NETKEIBA_THROTTLE_MIN=[0-9.]* NETKEIBA_THROTTLE_MAX=[0-9.]*')"; else echo "既定（1.8〜2.2 秒）"; fi; }
case "$ACTION" in
  status) status; exit 0 ;;
  apply|revert)
    if [ -z "$FILE" ]; then d="$(dirname "$(readlink -f "$0")")/../.."; mkdir -p "$d/data/local/meta/cron_backups"; printf '%s\n' "$CUR" > "$d/data/local/meta/cron_backups/crontab_before_fetch_${ACTION}_$(date +%Y%m%d_%H%M%S).txt"; fi
    NEW="$(printf '%s\n' "$CUR" | python3 -c "
import re, sys
mark, tag, action = sys.argv[1], sys.argv[2], sys.argv[3]
out = []
for ln in sys.stdin.read().splitlines():
    if tag in ln and 'backfill_past_night.sh' in ln:
        ln = re.sub(r'env NETKEIBA_THROTTLE_MIN=[0-9.]+ NETKEIBA_THROTTLE_MAX=[0-9.]+ ', '', ln)
        if action == 'apply':
            ln = re.sub(r'-- bash (\S*backfill_past_night\.sh)', lambda m: '-- env ' + mark + 'bash ' + m.group(1), ln, count=1)
    out.append(ln)
print('\n'.join(out))
" "$MARK" "$TAG" "$ACTION")"
    printf '%s\n' "$NEW" | write_cron
    CUR="$NEW"; LINE="$(printf '%s\n' "$CUR" | grep -F -- "$TAG" | grep -F 'backfill_past_night.sh')"; status ;;
  *) echo "使い方: status | apply | revert" >&2; exit 2 ;;
esac
