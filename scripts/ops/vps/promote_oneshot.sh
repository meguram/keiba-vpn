#!/usr/bin/env bash
# stg で確認した版を、開発 PC から 1 コマンドで prod（ConoHa VPS）へ昇格する。
#   bash scripts/ops/vps/promote_oneshot.sh [--commit <rev>] [--dry-run] [--yes] [--allow-dirty]
# 流れ: (1) 手元のコミットを確認（frontend の未コミット変更があれば中止）
#       (2) そのコミットをホストの refs/promote/ui-<日時> と ui-latest に push（ホストの作業ツリーは触らない）
#       (3) ホストで promote.sh（前検査: stg の UI ソース・フラグとコミットが同じか → ビルド → VPS へ転送 → 導入。失敗なら自動で戻る）
#       (4) 公開 URL の /version.json で反映を確認
# 戻し方: ホストで  sudo -n bash scripts/ops/vps/rollback.sh  （詳細は docs/operations/ の VPS 運用）
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"; cd "$ROOT"
HOST="${PROMOTE_HOST:-keiba-remote}"; HOST_REPO="${PROMOTE_HOST_REPO:-/home/jovyan/work/keiba-vpn}"
VPS_ALIAS="${PROMOTE_VPS_ALIAS:-keiba-vps}"; PUBLIC_URL="${PROMOTE_PUBLIC_URL:-https://app.megu-keiba.com}"
REV=HEAD; DRY=0; YES=0; DIRTY_OK=0
while [ $# -gt 0 ]; do case "$1" in
  --commit) REV="$2"; shift 2;; --dry-run) DRY=1; shift;; --yes) YES=1; shift;; --allow-dirty) DIRTY_OK=1; shift;;
  *) echo "不明な引数: $1" >&2; exit 1;; esac; done

COMMIT="$(git rev-parse --verify "${REV}^{commit}")"
echo "昇格するコミット: $COMMIT"; git log -1 --format='  %h %s' "$COMMIT"

if [ "$DIRTY_OK" != 1 ] && [ -n "$(git status --porcelain -- frontend config/ports.env)" ]; then
  echo "中止: frontend に未コミットの変更があります（stg に出した内容と prod のコミットがずれます）。先にコミットしてください。" >&2
  git status --short -- frontend config/ports.env | head -10 >&2; exit 1
fi

if [ "$YES" != 1 ] && [ "$DRY" != 1 ]; then read -r -p "prod に反映しますか？ [y/N] " a; [ "$a" = y ] || { echo "中止"; exit 1; }; fi

STAMP="$(date +%Y%m%d%H%M%S)"
echo "== 1/3 コミットをホストへ送る（refs/promote/ui-latest）"
git push -f --receive-pack='sudo -n git-receive-pack' "$HOST:$HOST_REPO" \
  "$COMMIT:refs/promote/ui-$STAMP" "$COMMIT:refs/promote/ui-latest"

echo "== 2/3 ホストで promote.sh を実行"
FLAGS="--commit $COMMIT --ssh $VPS_ALIAS --yes"; [ "$DRY" = 1 ] && FLAGS="--commit $COMMIT --ssh $VPS_ALIAS --dry-run"
ssh "$HOST" "cd $HOST_REPO && sudo -n env PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin bash scripts/ops/vps/promote.sh $FLAGS"

if [ "$DRY" = 1 ]; then echo "(dry-run) 反映はしていません"; exit 0; fi
echo "== 3/3 公開 URL の版を確認"
sleep 5; curl -s -m 15 "$PUBLIC_URL/version.json" || true; echo
echo "完了。stg との照合や戻し方は docs/operations/ の VPS 運用を参照。"
