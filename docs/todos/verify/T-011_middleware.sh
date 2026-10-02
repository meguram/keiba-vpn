#!/usr/bin/env bash
# T-011: Next.js middleware が開発者限定ページだけを保護し、ゲスト向けページを閉じないこと
# Next.js の開発サーバーを一時ポートで起動して curl で検証する（API サーバーは不要。ページ本体の表示は検証しない）。
source "$(dirname "$0")/_lib.sh"
TITLE="T-011 Next.js middleware"

echo "$TITLE"
cd "$ROOT/frontend" || exit 1
[ -d node_modules ] || { fail "frontend/node_modules が無い（cd frontend && npm ci を先に実行）"; finish; exit 1; }

check "型検査: tsc --noEmit" npx --no-install tsc --noEmit -p .
check "lint: middleware.ts" npx --no-install next lint --file middleware.ts

PORT="${FRONT_PORT:-$(free_port)}"
KEIBA_ENV=dev npx --no-install next dev -p "$PORT" >"$TMP_DIR/next.log" 2>&1 &
BG_PIDS+=($!)
info "Next.js 開発サーバー起動中 (port $PORT) …"
if ! wait_http "http://127.0.0.1:$PORT/login" 180; then
  fail "Next.js が起動しなかった"; tail -n 20 "$TMP_DIR/next.log" | sed 's/^/         | /'; finish; exit 1
fi

# status_and_location PATH [Cookie] → "コード Location"
probe() {
  local path="$1" cookie="${2:-}"
  if [ -n "$cookie" ]; then
    curl -s -o /dev/null --max-time 120 -H "Cookie: $cookie" -w '%{http_code} %{redirect_url}' "http://127.0.0.1:$PORT$path"
  else
    curl -s -o /dev/null --max-time 120 -w '%{http_code} %{redirect_url}' "http://127.0.0.1:$PORT$path"
  fi
}
expect_redirect_to_login() {
  local desc="$1" path="$2" out; out="$(probe "$path")"
  if [[ "$out" =~ ^30[1-8]\ .*/login\?next= ]]; then pass "$desc → $out"; else fail "$desc（期待: /login?next= へリダイレクト）実際: $out"; fi
}
expect_no_login_redirect() {
  local desc="$1" path="$2" cookie="${3:-}" out; out="$(probe "$path" "$cookie")"
  if [[ "$out" =~ /login ]]; then fail "$desc（/login へ飛ばされた）実際: $out"; else pass "$desc → ${out%% *}"; fi
}

echo "-- Cookie なし: 開発者限定ページは /login へ"
for p in /betting /betting/x /track-speed/dev /monitor /data-viewer /queue-status /server-logs /scrape-upcoming; do
  expect_redirect_to_login "GET $p" "$p"
done
echo "-- Cookie なし: ゲスト向けページは閉じない（旧実装の前方一致の取りこぼしと、過剰保護の両方を確認）"
for p in / /races /login /bloodline /megu-index /race/202606010101 /horse/2020100001 /betting-simulation /track-speed /myostatin; do
  expect_no_login_redirect "GET $p" "$p"
done
echo "-- Cookie あり（値は検証しない仕様）: 開発者限定ページも通る"
expect_no_login_redirect "GET /betting + keiba_dev_session" /betting "keiba_dev_session=dummy"
expect_no_login_redirect "GET /track-speed/dev + keiba_dev_session" /track-speed/dev "keiba_dev_session=dummy"
finish
