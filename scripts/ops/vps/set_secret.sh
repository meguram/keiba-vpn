#!/usr/bin/env bash
# 秘密の値を標準入力から受け取り、ファイルの「NAME=値」の行に書く（値は画面・ログ・ps に出さない。末尾の改行を必ず付ける）。
# 使い方（値は引数にしない）:
#   printf '%s' '<値>' | sudo bash set_secret.sh /etc/keiba-ui/access.env CF_ACCESS_CLIENT_ID
#   printf '%s' '<画面のコマンド全体でも可>' | sudo bash set_secret.sh /etc/cloudflared/prod.env TUNNEL_TOKEN   # eyJ… だけを自動で抜き出す
set -euo pipefail
FILE="${1:?ファイルのパス}"; NAME="${2:?変数名}"
umask 077
IN="$(cat)"
if [ "$NAME" = "TUNNEL_TOKEN" ]; then
  VAL="$(printf '%s' "$IN" | grep -o 'eyJ[A-Za-z0-9_=-]*' | head -n1 || true)"
else
  VAL="$(printf '%s' "$IN" | tr -d '\r\n' | sed -E "s/^[[:space:]]+|[[:space:]]+$//g; s/^${NAME}=//")"
fi
[ -n "$VAL" ] || { echo "値を読み取れません（空）" >&2; exit 2; }
if [ "$NAME" = "TUNNEL_TOKEN" ] && [ "${#VAL}" -lt 100 ]; then echo "トークンが短すぎます（eyJ… で始まる値を渡してください）" >&2; exit 2; fi
mkdir -p "$(dirname "$FILE")"
touch "$FILE"; chmod 600 "$FILE"
# 末尾に改行が無いファイルに追記して行がつながる事故を防ぐ
if [ -s "$FILE" ] && [ "$(tail -c1 "$FILE" | xxd -p)" != "0a" ]; then printf '\n' >> "$FILE"; fi
TMP="$(mktemp)"; trap 'rm -f "$TMP"' EXIT
grep -v "^${NAME}=" "$FILE" > "$TMP" || true
printf '%s=%s\n' "$NAME" "$VAL" >> "$TMP"
cat "$TMP" > "$FILE"
echo "更新しました: ${FILE} の ${NAME}（長さ ${#VAL}）"
