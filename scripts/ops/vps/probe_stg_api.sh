#!/usr/bin/env bash
# stg の画面が使う API を、VPS の nginx(8081 = Cloud Run の stg タグ付きリビジョン)に対して一通り叩き、取得できないものを一覧にする。値は表示しない。
#   bash probe_stg_api.sh [ベースURL] [10/24 のレース ID] [10/25 のレース ID]
# ログインは test/test（stg のリビジョンだけで有効）。本番(8080)に向けるときは第 1 引数で指定し、ログインはできないので匿名の確認になる。
set -u
B="${1:-http://127.0.0.1:8081}"; R1="${2:-202608040801}"; R2="${3:-202608040901}"; D1=20261024; D2=20261025
CJ="$(mktemp)"; trap 'rm -f "$CJ"' EXIT
code=$(curl -s -m 20 -c "$CJ" -X POST "$B/api/v1/auth/login" -H 'Content-Type: application/json' -d '{"id":"test","password":"test","remember":false}' -o /dev/null -w '%{http_code}')
echo "ログイン(test): $code  ※200 以外だと以下は匿名の結果"
n=0; bad=0
probe() {
  local url="$1" out c
  out="$(mktemp)"; c=$(curl -s -m 40 -b "$CJ" -o "$out" -w '%{http_code}' "$B$url"); n=$((n+1))
  if [ "$c" != 200 ]; then bad=$((bad+1)); printf '%-4s %s\n     %s\n' "$c" "$url" "$(head -c 140 "$out" | tr '\n' ' ')"; fi
  rm -f "$out"
}
for r in "$R1" "$R2"; do
  probe "/api/race/$r"; probe "/api/race/$r/predictions"; probe "/api/race/$r/tracking-difficulty"; probe "/api/v1/races/$r/megu-index-predicted"
done
for d in $D1 $D2; do probe "/api/race-list/$d"; probe "/api/v1/betting/race-values?date=$d"; probe "/api/prediction-results/bets-summary?date=$d"; probe "/api/prediction-results?scope=day&date=$d"; done
probe "/api/scrape-dates?picker_past_days=14&picker_future_days=14"; probe "/api/race-calendar?month=202610"
probe "/api/prediction-results/days"; probe "/api/prediction-results/mark-card"; probe "/api/prediction-results/news?limit=6"; probe "/api/prediction-results/bets-by-level"
probe "/api/maintenance/status"; probe "/api/v1/auth/status"
echo "確認 $n 件、200 以外 $bad 件"
