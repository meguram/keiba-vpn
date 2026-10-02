#!/usr/bin/env bash
# 学習PC検証スクリプトの共通部品。各 T-xxx スクリプトから source される。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1

PASS=0; FAIL=0; SKIP=0
BG_PIDS=()
TMP_DIR="$(mktemp -d /tmp/keiba-verify.XXXXXX)"

pass() { echo "  [PASS] $*"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $*"; FAIL=$((FAIL + 1)); }
skip() { echo "  [SKIP] $*"; SKIP=$((SKIP + 1)); }
info() { echo "  [INFO] $*"; }

# check "説明" コマンド... : 終了コード 0 なら PASS。失敗時は出力の末尾を表示する
check() {
  local desc="$1"; shift
  local log="$TMP_DIR/check.$$.log"
  if "$@" >"$log" 2>&1; then
    pass "$desc"
  else
    fail "$desc"; tail -n 15 "$log" | sed 's/^/         | /'
  fi
}

# py_check "説明" [引数...] : 標準入力の Python を実行。終了コード 0=PASS / 3=SKIP（前提が無い）/ その他=FAIL
py_check() {
  local desc="$1"; shift
  python3 - "$@"
  local rc=$?
  case "$rc" in
    0) pass "$desc" ;;
    3) skip "$desc（前提が無いためスキップ）" ;;
    *) fail "$desc" ;;
  esac
}

# py_report [引数...] : 標準入力の Python を実行。行頭 "  [PASS]" / "  [FAIL]" / "  [SKIP]" の行を数えて集計に加える
py_report() {
  local out rc
  out="$(python3 - "$@" 2>&1)"; rc=$?
  echo "$out"
  PASS=$((PASS + $(grep -c '^  \[PASS\]' <<<"$out" || true)))
  FAIL=$((FAIL + $(grep -c '^  \[FAIL\]' <<<"$out" || true)))
  SKIP=$((SKIP + $(grep -c '^  \[SKIP\]' <<<"$out" || true)))
  if [ "$rc" -ne 0 ] && ! grep -q '^  \[FAIL\]' <<<"$out"; then fail "Python が異常終了（終了コード $rc）"; fi
}

# .env / .env.<KEIBA_ENV> の値を取り出す（値はログに出さず、呼び出し側が変数に入れる）
env_get() {
  python3 - "$1" <<'PY'
import os, sys
try:
    from src.utils.project_env import load_project_dotenv
    load_project_dotenv()
except Exception:
    pass
print(os.environ.get(sys.argv[1], ""))
PY
}

# wait_http URL 秒数 : 200 系/4xx が返るまで待つ（接続できれば起動済みとみなす）
wait_http() {
  local url="$1" limit="${2:-90}" i=0
  while [ "$i" -lt "$limit" ]; do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$url" || true)"
    [ "${code:-000}" != "000" ] && return 0
    sleep 1; i=$((i + 1))
  done
  return 1
}

free_port() {
  python3 - <<'PY'
import socket
s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()
PY
}

cleanup() {
  for pid in "${BG_PIDS[@]:-}"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null
  done
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

finish() {
  echo "== ${TITLE:-verify}: PASS=$PASS FAIL=$FAIL SKIP=$SKIP"
  [ "$FAIL" -eq 0 ]
}
