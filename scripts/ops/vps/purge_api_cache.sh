#!/usr/bin/env bash
# VPS の API キャッシュ（nginx proxy_cache keiba_apicache）を空にする（root 権限）。
# 確定済みのレースは 1 か月置くので、データ更新・モデル切替・デプロイのあとはここで入れ替える。
#   sudo bash scripts/ops/vps/purge_api_cache.sh                         全部消す
#   sudo bash scripts/ops/vps/purge_api_cache.sh --match '/api/race/2026' キー(ポート+ホスト+URL)が正規表現(ERE)に合うものだけ消す
#   sudo bash scripts/ops/vps/purge_api_cache.sh --match '^8081'          stg だけ消す（本番は 8080）
#   末尾にキャッシュディレクトリを渡せる（既定 /var/cache/nginx/keiba_api）。
# 消したあと最初の閲覧が上流に取りに行き、自動で入れ直される。
set -euo pipefail
MATCH=""; DIR=/var/cache/nginx/keiba_api
while [ $# -gt 0 ]; do
  case "$1" in
    --match) [ $# -ge 2 ] || { echo "--match に正規表現を渡してください" >&2; exit 2; }; MATCH="$2"; shift 2 ;;
    *) DIR="$1"; shift ;;
  esac
done
if [[ ! -d "$DIR" ]]; then echo "キャッシュディレクトリが見つかりません: $DIR" >&2; exit 1; fi
if [ -z "$MATCH" ]; then
  N=$(find "$DIR" -type f | wc -l)
  find "$DIR" -type f -delete
else
  # キャッシュファイルの先頭付近に "KEY: <キー>" の行がある
  N=0
  # 先頭が ^ ならキーの先頭に合わせる（"KEY: " の直後）
  if [ "${MATCH:0:1}" = "^" ]; then PAT="^KEY: (${MATCH#^})"; else PAT="^KEY: .*(${MATCH})"; fi
  while IFS= read -r -d '' f; do rm -f -- "$f"; N=$((N+1)); done < <(grep -a -r -l -Z -E "$PAT" "$DIR" || true)
fi
nginx -s reload
echo "API キャッシュを消しました: ${N} 件 ($DIR${MATCH:+, 条件: $MATCH})"
