#!/usr/bin/env bash
# VPS: ホストが作った API の応答（src/ops/edge_render.py）をフォルダから返す設定を入れる（root で）。docs/operations/edge_cache_plan_20261121.md §10。
#
#   sudo bash scripts/ops/vps/vps_14_edge_files.sh --env stg        # 置き場・スニペット・サイトの include を入れる（stg の 8081 だけ）
#   sudo bash scripts/ops/vps/vps_14_edge_files.sh --env prod
#   sudo bash scripts/ops/vps/vps_14_edge_files.sh --env stg --remove
#
# - 置き場: /srv/keiba-edge/<env>/（deploy が書き、nginx（www-data）が読む）。_ctl/ は版の置き場の外（rsync の --delete で消えない）。
# - スニペット: /etc/nginx/snippets/keiba-edge-<env>.conf（templates/nginx-edge-files.conf）。
# - サイト: その env の server ブロックの `location ~ ^/api/` の直前に include の 1 行を足す（2 回目は何もしない）。
# - 変える前に退避し、nginx -t が落ちたら戻して止まる。
# - 止め方（reload なし）: touch /srv/keiba-edge/<env>/_ctl/OFF（deploy で置ける）。
set -euo pipefail

ENV_NAME=""; REMOVE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --remove) REMOVE=1; shift ;;
    *) echo "不明な引数: $1" >&2; exit 2 ;;
  esac
done
case "$ENV_NAME" in prod) SITE=/etc/nginx/sites-available/keiba-ui ;; stg) SITE=/etc/nginx/sites-available/keiba-ui-stg ;; *) echo "--env は prod か stg" >&2; exit 2 ;; esac
[ "$(id -u)" = 0 ] || { echo "root で流す" >&2; exit 2; }

HERE="$(cd "$(dirname "$0")" && pwd)"
SNIP=/etc/nginx/snippets/keiba-edge-${ENV_NAME}.conf
ROOTD=/srv/keiba-edge
INC="    include /etc/nginx/snippets/keiba-edge-${ENV_NAME}*.conf;   # ホストが作った API の応答をフォルダから返す（vps_14_edge_files.sh）"
BK="${BK_ROOT:-/var/backups/keiba-nginx}/edge-files-${ENV_NAME}-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BK"
cp -a "$SITE" "$BK/site"
[ -f "$SNIP" ] && cp -a "$SNIP" "$BK/snippet" || true

restore() {
  echo "nginx -t が落ちたので戻します" >&2
  cp -a "$BK/site" "$SITE"
  if [ -f "$BK/snippet" ]; then cp -a "$BK/snippet" "$SNIP"; else rm -f "$SNIP"; fi
  nginx -t >/dev/null 2>&1 && systemctl reload nginx || true
}

if [ "$REMOVE" = 1 ]; then
  python3 - "$SITE" <<'PY'
import sys
p = sys.argv[1]
lines = open(p, encoding="utf-8").read().splitlines(keepends=True)
open(p, "w", encoding="utf-8").writelines(l for l in lines if "/etc/nginx/snippets/keiba-edge-" not in l)
PY
  rm -f "$SNIP"
  # 止めの印も置く（cache-warm はこれを見て、フォルダの代わりに nginx の API キャッシュの取り直しに戻る）
  [ -d "$ROOTD/$ENV_NAME/_ctl" ] && install -o deploy -g www-data -m 640 /dev/null "$ROOTD/$ENV_NAME/_ctl/OFF" || true
  if nginx -t; then systemctl reload nginx; echo "フォルダ配信を外しました（置き場は残す。バックアップ: $BK）"; else restore; exit 1; fi
  exit 0
fi

# サイトの形を先に確かめる（`location ~ ^/api/ {` がちょうど 1 つ。include が入っていれば何もしない）。スニペット・置き場を作る前に止める
if ! grep -q "/etc/nginx/snippets/keiba-edge-" "$SITE"; then
  n="$(grep -cE '^[[:space:]]*location ~ \^/api/ \{' "$SITE" || true)"
  [ "$n" = 1 ] || { echo "$SITE の /api/ の location が 1 つではない（${n} 個）。形が想定と違うので止める" >&2; exit 1; }
fi

# 置き場（deploy が書き、www-data が読む。setgid で新しいファイルも www-data の組）
install -d -o deploy -g www-data -m 2750 "$ROOTD" "$ROOTD/$ENV_NAME" "$ROOTD/$ENV_NAME/_ctl"
sed "s/__EDGE_ENV__/${ENV_NAME}/g" "$HERE/templates/nginx-edge-files.conf" > "$SNIP.tmp"
install -o root -g root -m 644 "$SNIP.tmp" "$SNIP"
rm -f "$SNIP.tmp"

# サイト: `location ~ ^/api/ {` の直前に include を足す（無ければ止まる。2 回目は何もしない）
python3 - "$SITE" "$INC" <<'PY'
import sys
p, inc = sys.argv[1], sys.argv[2]
text = open(p, encoding="utf-8").read()
if "/etc/nginx/snippets/keiba-edge-" in text:
    print("サイトには include が入っている（変えない）")
    sys.exit(0)
lines = text.splitlines(keepends=True)
idx = [i for i, l in enumerate(lines) if l.strip().startswith("location ~ ^/api/ {")]
if len(idx) != 1:
    sys.exit(f"`location ~ ^/api/ {{` が 1 つではない（{len(idx)} 個）。サイトの形が想定と違うので止める")
lines.insert(idx[0], inc + "\n")
open(p, "w", encoding="utf-8").writelines(lines)
print("サイトに include を足した")
PY

if nginx -t; then systemctl reload nginx; echo "フォルダ配信を入れました（${ENV_NAME}。バックアップ: $BK）"; else restore; exit 1; fi
