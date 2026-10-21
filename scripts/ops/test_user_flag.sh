#!/usr/bin/env bash
# テストユーザー（id: test / pw: test。config/members.json にはハッシュだけ）の有効・無効を .env の KEIBA_TEST_USER で切り替える。
# .env を書き換えるだけで再起動はしない（反映は API の次の再起動＝メンテナンス窓の自動デプロイ、または手動の restart_server.sh）。
#   bash scripts/ops/test_user_flag.sh on      # stg 用: KEIBA_TEST_USER=1
#   bash scripts/ops/test_user_flag.sh off     # 公開前・prod: KEIBA_TEST_USER=0
#   bash scripts/ops/test_user_flag.sh status
# 公開環境（誰でも開ける状態）では off のままにすること。test/test は誰でも推測できる。
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
ENVF="${ENV_FILE:-$ROOT/.env}"
ACTION="${1:-status}"
[ -f "$ENVF" ] || { echo "$ENVF がありません" >&2; exit 1; }
cur="$(grep -E '^KEIBA_TEST_USER=' "$ENVF" | tail -n1 | cut -d= -f2- || true)"
case "$ACTION" in
  status) echo "KEIBA_TEST_USER=${cur:-（未設定=無効）}" ;;
  on|off)
    val=1; [ "$ACTION" = off ] && val=0
    cp -p "$ENVF" "$ENVF.bak.$(date +%Y%m%d%H%M%S)"
    if grep -qE '^KEIBA_TEST_USER=' "$ENVF"; then
      sed -i -E "s/^KEIBA_TEST_USER=.*/KEIBA_TEST_USER=$val/" "$ENVF"
    else
      [ -n "$(tail -c1 "$ENVF")" ] && printf '\n' >> "$ENVF"    # 末尾に改行が無いと前の行とくっつく
      printf 'KEIBA_TEST_USER=%s\n' "$val" >> "$ENVF"
    fi
    echo "KEIBA_TEST_USER=$val に設定しました（再起動後に反映）" ;;
  *) echo "Usage: $0 on|off|status" >&2; exit 2 ;;
esac
