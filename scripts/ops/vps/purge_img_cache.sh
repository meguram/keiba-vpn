#!/usr/bin/env bash
# VPS の画像キャッシュ（nginx proxy_cache keiba_img）を空にする。ホスト側の画像を差し替えたときに実行する（root 権限）。
# 使い方: sudo bash scripts/ops/vps/purge_img_cache.sh [キャッシュディレクトリ]
set -euo pipefail
DIR="${1:-/var/cache/nginx/keiba_img}"
if [[ ! -d "$DIR" ]]; then echo "キャッシュディレクトリが見つかりません: $DIR" >&2; exit 1; fi
find "$DIR" -type f -delete
nginx -s reload
echo "画像キャッシュを空にしました: $DIR"
