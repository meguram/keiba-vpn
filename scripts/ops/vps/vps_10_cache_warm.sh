#!/usr/bin/env bash
# API キャッシュの事前ロード係（cache_warm.py + systemd タイマー 60 秒おき）を入れる/外す(冪等。root で実行。vps_09 と vps_03 --with-pull-venv の後)。
#   sudo bash vps_10_cache_warm.sh [--dry-run]     # 入れる
#   sudo bash vps_10_cache_warm.sh --remove        # 外す
# 何をするか: GCS の ops/cache_dirty/ を 60 秒に 1 回 list し（read 課金は月 約 0.2 ドル）、参照用データが更新されたレース/開催日の API を
#   ローカル nginx(127.0.0.1:8080)へ X-Keiba-Refresh 付きで取り直してキャッシュに入れる。キャッシュ本体は VPS ローカル。保持は API の Cache-Control(確定レース 30 日)。
# 前提: /etc/keiba-ui/refresh.token（vps_09）、/etc/keiba-pull/pull.env と /opt/keiba-pull/venv（vps_03_units.sh --with-pull-venv）。
#       読み取り専用 SA が ops/cache_dirty/ を list できること（bucket 全体の objectViewer なら足りる）。
# 任意: 目印は 1 日 1 接頭辞に溜まるだけで小さい。気になるなら GCS のライフサイクルで ops/cache_dirty/ を 7 日で削除してよい。
set -euo pipefail
DRY=0; REMOVE=0
for a in "$@"; do case "$a" in --dry-run) DRY=1;; --remove) REMOVE=1;; *) echo "不明な引数: $a" >&2; exit 2;; esac; done
[ "$(id -u)" = 0 ] || [ "$DRY" = 1 ] || { echo "root で実行してください" >&2; exit 2; }
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
run() { printf '+'; printf ' %q' "$@"; printf '\n'; [ "$DRY" = 1 ] || "$@"; }
if [ "$REMOVE" = 1 ]; then
  run systemctl disable --now keiba-cache-warm.timer || true
  run rm -f /etc/systemd/system/keiba-cache-warm.service /etc/systemd/system/keiba-cache-warm.timer /opt/keiba-ui/bin/cache_warm.py
  run systemctl daemon-reload
  echo "外しました(状態 /var/lib/keiba-ui/cache-warm は残しています)"; exit 0
fi
run install -d -m 755 /opt/keiba-ui/bin
run install -d -m 700 /var/lib/keiba-ui/cache-warm
run install -m 755 -o root -g root "$HERE/cache_warm.py" /opt/keiba-ui/bin/cache_warm.py
run install -m 644 "$HERE/templates/keiba-cache-warm.service" "$HERE/templates/keiba-cache-warm.timer" /etc/systemd/system/
run systemctl daemon-reload
if [ "$DRY" = 1 ]; then echo "(dry-run) 変更は何もしていません"; exit 0; fi
missing=""
[ -s /etc/keiba-ui/refresh.token ] || missing="$missing refresh.token(vps_09_api_cache.sh)"
grep -q '^GCS_BUCKET=.' /etc/keiba-pull/pull.env 2>/dev/null || missing="$missing pull.env(vps_03_units.sh --with-pull-venv)"
/opt/keiba-pull/venv/bin/python -c 'import google.cloud.storage' 2>/dev/null || missing="$missing pull用venvのgoogle-cloud-storage"
if [ -z "$missing" ]; then
  (set -a; . /etc/keiba-pull/pull.env; set +a; exec /opt/keiba-pull/venv/bin/python /opt/keiba-ui/bin/cache_warm.py --dry-run) || { echo "動作確認(--dry-run)に失敗しました。タイマーは未有効" >&2; exit 1; }
  systemctl enable --now keiba-cache-warm.timer
  echo "有効にしました。確認: journalctl -u keiba-cache-warm -n 5 / systemctl list-timers keiba-cache-warm.timer"
else
  echo "前提が未完了です(タイマーは未有効):$missing"
  echo "そろったら: sudo systemctl enable --now keiba-cache-warm.timer"
fi
