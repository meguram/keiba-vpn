#!/usr/bin/env bash
# nginx に、応答時間とキャッシュの状態を残す 2 本目のアクセスログを入れる/外す。root で実行。冪等・失敗したら元に戻す。
#   sudo bash scripts/ops/vps/vps_13_access_log.sh [--dry-run|--remove]
# 入れる物: /etc/nginx/conf.d/keiba-log.conf（templates/nginx-keiba-log.conf）・/var/log/nginx/keiba/（deploy が読める）・
#          /etc/logrotate.d/keiba-nginx-timed（templates/logrotate-keiba-nginx-timed。14 日・圧縮）
# 読み方: tail -f /var/log/nginx/keiba/timed.log（deploy で sudo なし）。例: 本番の API の遅いもの
#   grep ' port=8080 ' /var/log/nginx/keiba/timed.log | grep '"GET /api/' | awk '{for(i=1;i<=NF;i++) if($i ~ /^rt=/) {sub("rt=","",$i); if ($i+0 > 1) print}}'
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NGX="${NGX_ETC:-/etc/nginx}"
CONF="$NGX/conf.d/keiba-log.conf"
ROT="/etc/logrotate.d/keiba-nginx-timed"
LOGDIR="/var/log/nginx/keiba"
LOG_GROUP="${LOG_GROUP:-deploy}"
MODE="${1:-apply}"
if [ "$MODE" = "--dry-run" ]; then
  echo "(dry-run) 追加: $CONF / $ROT / $LOGDIR（グループ $LOG_GROUP が読める）"; exit 0
fi
BK="${BK_ROOT:-/var/backups/keiba-nginx}/access-log-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BK"
HAD_CONF=0; [ -f "$CONF" ] && { cp -p "$CONF" "$BK/keiba-log.conf"; HAD_CONF=1; }
HAD_ROT=0; [ -f "$ROT" ] && { cp -p "$ROT" "$BK/keiba-nginx-timed"; HAD_ROT=1; }
restore() {   # 設定 2 つを入れる前の状態へ戻す（ログの置き場 $LOGDIR は空でも害が無いので残す）
  if [ "$HAD_CONF" = 1 ]; then cp -p "$BK/keiba-log.conf" "$CONF"; else rm -f "$CONF"; fi
  if [ "$HAD_ROT" = 1 ]; then cp -p "$BK/keiba-nginx-timed" "$ROT"; else rm -f "$ROT"; fi
  nginx -t >/dev/null 2>&1 && systemctl reload nginx || true
  echo "元に戻しました: $BK" >&2
}
if [ "$MODE" = "--remove" ]; then
  rm -f "$CONF" "$ROT"
  if nginx -t; then systemctl reload nginx; echo "外しました（ログ $LOGDIR は残しています。バックアップ: $BK）"; else restore; exit 1; fi
  exit 0
fi
for f in nginx-keiba-log.conf logrotate-keiba-nginx-timed; do
  [ -f "$HERE/templates/$f" ] || { echo "templates/$f がありません" >&2; exit 1; }
done
getent group "$LOG_GROUP" >/dev/null || { echo "グループ $LOG_GROUP がありません" >&2; exit 1; }
# 持ち主は www-data（ログの開き直し USR1 で worker が開く）。グループ（deploy）は読むだけ。グループ・他人は書けない（logrotate の安全確認も通る）
install -d -m 750 -o www-data -g "$LOG_GROUP" "$LOGDIR"
[ -f "$LOGDIR/timed.log" ] || install -m 640 -o www-data -g "$LOG_GROUP" /dev/null "$LOGDIR/timed.log"
sed "s/create 0640 www-data deploy/create 0640 www-data $LOG_GROUP/" "$HERE/templates/logrotate-keiba-nginx-timed" > "$ROT.new"
chmod 644 "$ROT.new" && mv "$ROT.new" "$ROT"
logrotate -d "$ROT" >/dev/null 2>&1 || { echo "logrotate の設定が読めません: $ROT" >&2; restore; exit 1; }
install -m 644 "$HERE/templates/nginx-keiba-log.conf" "$CONF"
if nginx -t; then systemctl reload nginx; echo "応答時間つきのログを有効にしました: $LOGDIR/timed.log（バックアップ: $BK）"; else restore; exit 1; fi
