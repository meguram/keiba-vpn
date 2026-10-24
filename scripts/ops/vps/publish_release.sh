#!/usr/bin/env bash
# 【pull 型の昇格】stg で確認した版を GCS に置く。VPS はメンテナンス窓（JST 03:00〜04:00）に自分で取りに行って入れ替える（SSH 不要）。
#   bash scripts/ops/vps/publish_release.sh [--commit <rev>] [--yes] [--dry-run] [--skip-parity] [--fetch]
# promote.sh（SSH で即時に入れる）と共存する。急ぐとき・窓の外は promote.sh、普段はこちら。latest.json を書く＝『prod に出す』という人の判断。
#   --fetch  UI ではなく、取得の実行役のコードバンドル（src/ と取得が読む config/）を配布する
# ホスト機で実行（GCS の資格情報は .env）。VPS 側の適用結果: python -m src.ops.release_publish status [--kind fetch]
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"; cd "$ROOT"
REV=HEAD; YES=0; DRY=0; SKIP_PARITY=0; FETCH=0
while [ $# -gt 0 ]; do case "$1" in --commit) REV="$2"; shift 2;; --yes) YES=1; shift;; --dry-run) DRY=1; shift;; --skip-parity) SKIP_PARITY=1; shift;; --fetch) FETCH=1; shift;; *) echo "不明な引数: $1" >&2; exit 1;; esac; done
COMMIT="$(git rev-parse --verify "${REV}^{commit}")"
KIND=ui; [ "$FETCH" = 1 ] && KIND=fetch
echo "配布するコミット: $COMMIT  種別: $KIND  → GCS ops/releases/$KIND/（VPS が次の窓で取得）"; git log -1 --format='  %h %s' "$COMMIT"
if [ "$YES" != 1 ] && [ "$DRY" != 1 ]; then read -r -p "prod の次のメンテナンス窓に反映する版として公開しますか？ [y/N] " a; [ "$a" = y ] || { echo "中止"; exit 1; }; fi
if [ "$KIND" = ui ] && [ "$SKIP_PARITY" != 1 ]; then PYTHONPATH="$ROOT" python3 -m src.scripts.ops.env_parity_check promote-precheck --commit "$COMMIT" || { echo "中止（承知の上なら --skip-parity）" >&2; exit 1; }; fi
if [ "$KIND" = ui ]; then
  [ "$DRY" = 1 ] && { echo "(dry-run) ビルド・アップロードはしません"; exit 0; }
  OUTFILE="$(bash "$ROOT/scripts/ops/vps/build_release.sh" --commit "$COMMIT" | tail -1)"
  PYTHONPATH="$ROOT" python3 -m src.ops.release_publish publish-ui --file "$OUTFILE" --commit "$COMMIT"
else
  [ "$DRY" = 1 ] && { echo "(dry-run) バンドルの作成・アップロードはしません"; exit 0; }
  OUTFILE="$(PYTHONPATH="$ROOT" python3 -m src.ops.release_publish build-fetch | tail -1)"
  PYTHONPATH="$ROOT" python3 -m src.ops.release_publish publish-fetch --file "$OUTFILE" --commit "$COMMIT"
fi
mkdir -p data/local/meta
printf '{"at":"%s","release":"%s","commit":"%s","result":"published","kind":"%s"}\n' "$(date -Is)" "$(basename "$OUTFILE" .tar.gz)" "$COMMIT" "$KIND" >> data/local/meta/vps_promote_history.jsonl
echo "公開しました。VPS は次の JST 03:10（UI）/ 03:20（取得の実行役）に取得します。確認: python3 -m src.ops.release_publish status --kind $KIND"
