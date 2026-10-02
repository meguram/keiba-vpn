#!/usr/bin/env bash
# 学習PCでの一括検証。各項目のスクリプトを順に実行し、最後に結果一覧を出す。
#
#   bash docs/todos/verify/run_all.sh                 # 全項目（サーバー起動を含む）
#   RACE_ID=202606010101 bash docs/todos/verify/run_all.sh   # 実データ確認も行う
#   SKIP_SERVERS=1 bash docs/todos/verify/run_all.sh  # FastAPI / Flask / Next.js を起動する項目を飛ばす
#
# ログは ~/keiba-verify-<日時>.log にも保存される（FAIL があればこのファイルを共有する）。
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HOME/keiba-verify-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee "$LOG") 2>&1

ITEMS=(T-060_settings T-013_harville T-029_t45_output T-055_circuit_breaker)
if [ "${SKIP_SERVERS:-0}" != "1" ]; then
  ITEMS+=(T-011_middleware T-010_auth T-032_betting)
fi

echo "検証開始: $(date '+%F %T')  host=$(hostname)  branch=$(git -C "$HERE" rev-parse --abbrev-ref HEAD)  commit=$(git -C "$HERE" rev-parse --short HEAD)"
echo "RACE_ID=${RACE_ID:-（未指定）}  SKIP_SERVERS=${SKIP_SERVERS:-0}"
echo

declare -A RESULT
for item in "${ITEMS[@]}"; do
  echo "################ $item"
  if bash "$HERE/$item.sh"; then RESULT[$item]="OK"; else RESULT[$item]="NG"; fi
  echo
done

echo "================ 結果一覧"
rc=0
for item in "${ITEMS[@]}"; do
  printf '  %-26s %s\n' "$item" "${RESULT[$item]}"
  [ "${RESULT[$item]}" = "OK" ] || rc=1
done
echo "ログ: $LOG"
exit $rc
