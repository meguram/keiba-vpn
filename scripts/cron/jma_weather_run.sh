#!/usr/bin/env bash
# 気象庁の時別値（風向・風速など）の補いを、1 日 1 回だけ行う（6 時 40 分以降の最初の回。PC が止まっていた朝も、起動後の回で拾う）。
# 範囲は「前回済んだ日の 2 日前 〜 昨日」（最大 31 日前から。済んだ記録が無ければ直近 3 日）。PC や印で止まっていた日も、次に済んだ回で埋まる。
# 失敗した日は 3 回まで（毎時）やり直し、それを超えたら終了コード 6 でその日は回さない（失敗の記録で見える）。
# cron は毎時起動のまま（crontab を変えない）。今日すでに終えていれば何もしない（GCS も読まない）。
# 費用: daily は 競馬場 10 × 月ファイル 1〜2 件（約 270KB）を読み、10 件を書く。1 日 約 2.7〜5.4MB → 月 約 80〜160MB ≒ 1.5〜3 円。
#   以前は毎時 `fetch --from 20160101` を回し、取得済みの判定に月ファイル 1,310 件（約 345MiB）を毎回読んでいた（月 約 4,400 円相当。2026-11-13 に発覚）。
#   2016 年以降の過去分の取得・空白の補いは手で 1 回だけ回す（1 回 約 345MiB ≒ 6 円。docs/operations/jma_weather.md）。
# エラー（HTTP が 200 以外・表が読めない）は 1 件で中断し、data/local/meta/jma_weather_halt.json を書く。確認して削除するまで再開しない。
# 仕様・運用: docs/operations/jma_weather.md   Usage: bash scripts/cron/jma_weather_run.sh [PROJECT_DIR]
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs data/local/meta
LOG="logs/jma_weather.log"; HALT="data/local/meta/jma_weather_halt.json"; DONE="data/local/meta/jma_weather_daily_done"; FAIL="data/local/meta/jma_weather_daily_fail"
PY="${PYTHON:-/opt/conda/bin/python3}"; MAX_FAIL="${JMA_DAILY_MAX_FAIL:-3}"
FIRST_HOUR="${JMA_DAILY_FIRST_HOUR:-6}"   # この時（JST）以降の最初の回で、その日の daily を行う
exec 9>"${JMA_LOCK:-/tmp/keiba_jma_weather.lock}"
flock -n 9 || exit 0
# 止めの印があるあいだは 5 で終わる（0 だと cron の記録で正常に見え、気付かれないまま止まり続ける。2026-10-26〜11-13 に 18 日間止まった）
if [ -e "$HALT" ]; then echo "[$(date -Is)] 中断印があるため始めません（確認して ${HALT} を削除）" >>"$LOG"; exit 5; fi
TODAY="$(TZ=Asia/Tokyo date +%F)"; HOUR="$(TZ=Asia/Tokyo date +%-H)"
LAST="$(cat "$DONE" 2>/dev/null)"
[ "$LAST" = "$TODAY" ] && exit 0                           # 今日は済んでいる
[ "$HOUR" -lt "$FIRST_HOUR" ] && exit 0                    # 前日分が出そろう朝まで待つ
FDAY=""; FN=0; [ -f "$FAIL" ] && read -r FDAY FN < "$FAIL"
[ "$FDAY" != "$TODAY" ] && FN=0
if [ "${FN:-0}" -ge "$MAX_FAIL" ]; then exit 6; fi        # 今日は上限まで失敗した（読み直しを繰り返さない）
# 起点: 前回済んだ日の 2 日前（31 日より前には戻らない）。記録が無ければ 3 日前
FROM="$(TZ=Asia/Tokyo date -d "$TODAY -3 day" +%Y%m%d)"
if [ -n "$LAST" ] && L2="$(TZ=Asia/Tokyo date -d "$LAST -2 day" +%Y%m%d 2>/dev/null)"; then
  MIN="$(TZ=Asia/Tokyo date -d "$TODAY -31 day" +%Y%m%d)"
  FROM="$L2"; [ "$FROM" -lt "$MIN" ] && FROM="$MIN"
fi
echo "===== $(date -Is) START from=${FROM} =====" >>"$LOG"
run() { KEIBA_ENV="${KEIBA_ENV:-stg}" PYTHONPATH=. nice -n 10 "$PY" -m src.scraper.jma_weather "$@" >>"$LOG" 2>&1; }
run fetch --from "$FROM"; RC=$?
if [ "$RC" -eq 0 ]; then echo "$TODAY" > "$DONE"; rm -f "$FAIL"; elif [ "$RC" -ne 3 ]; then echo "$TODAY $(( ${FN:-0} + 1 ))" > "$FAIL"; fi
[ "$RC" -eq 3 ] && printf '{"at":"%s","rc":%s,"log":"%s"}\n' "$(date -Is)" "$RC" "$LOG" > "$HALT"
echo "===== $(date -Is) END rc=${RC} =====" >>"$LOG"
exit "$RC"
