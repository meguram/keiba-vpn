#!/usr/bin/env bash
# （任意）旧サイトのデータを tar で退避する。ConoHa の『イメージ保存』の代わり/補助。VPS 上で管理ユーザーが実行（sudo のパスワードが要る）。
#   bash backup_old_site.sh [出力先ディレクトリ=$HOME/old_site_backup]
# 退避するもの: /var/www/megu-keiba（このリポジトリの古いコピー、uwsgi/nginx の設定、scripts）と $HOME。仮想環境(/opt/venv)は入れない（再作成できる）。
# 手元/ホストへ取り出す例（VPS には直接つなげない回線でもホスト経由なら届く）:
#   scp -o ProxyJump=keiba-remote rakuten-server:old_site_backup/*.tar.gz /path/to/backup/
set -euo pipefail
OUT="${1:-$HOME/old_site_backup}"; TS="$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT"
echo "空き容量:"; df -h "$OUT" | tail -1
echo "== /var/www/megu-keiba（venv・node_modules・キャッシュ・ログは除く）"
sudo tar --exclude='venv' --exclude='.venv' --exclude='node_modules' --exclude='__pycache__' --exclude='*.log' --exclude='.git/objects/pack' -czf "$OUT/var_www_megu-keiba_$TS.tar.gz" -C /var/www megu-keiba
echo "== $HOME（このバックアップ先と、ssh の秘密鍵は除く）"
tar --exclude="$(basename "$OUT")" --exclude='.ssh' --exclude='.cache' --exclude='node_modules' -czf "$OUT/home_$(whoami)_$TS.tar.gz" -C "$(dirname "$HOME")" "$(basename "$HOME")"
echo "== 設定（nginx / uwsgi / systemd / letsencrypt の設定。秘密鍵(privkey)は含めない）"
sudo tar --exclude='privkey*.pem' -czf "$OUT/etc_configs_$TS.tar.gz" /etc/nginx /etc/systemd/system/uwsgi.service /etc/systemd/system/keiba-api.service /etc/letsencrypt/renewal 2>/dev/null || true
sudo chown "$(id -u):$(id -g)" "$OUT"/*.tar.gz
( cd "$OUT" && sha256sum ./*"$TS".tar.gz > "SHA256SUMS_$TS" )
ls -lh "$OUT"/*"$TS"*
echo "完了。ホストへ取り出したら sha256sum -c SHA256SUMS_$TS で確認。取り出す前にこの VPS を削除・再構築しないこと。"
