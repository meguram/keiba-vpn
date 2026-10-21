#!/usr/bin/env bash
# stg で確認した版を prod（ConoHa VPS）へ昇格する。ホスト機か開発 PC で実行（SSH で VPS に入れること。ssh_config.example 参照）。
#   bash scripts/ops/vps/promote.sh [--commit <rev>] [--ssh keiba-vps] [--yes] [--dry-run] [--skip-parity]
# 流れ: ビルド → VPS の incoming へ転送 → install_release.sh（検証・起動テスト・切替・確認・失敗なら自動で戻す）→ バージョン記録。
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"; cd "$ROOT"
REV=HEAD; SSHH=keiba-vps; YES=0; DRY=0; SKIP_PARITY=0
while [ $# -gt 0 ]; do case "$1" in --commit) REV="$2"; shift 2;; --ssh) SSHH="$2"; shift 2;; --yes) YES=1; shift;; --dry-run) DRY=1; shift;; --skip-parity) SKIP_PARITY=1; shift;; *) echo "不明な引数: $1" >&2; exit 1;; esac; done
COMMIT="$(git rev-parse --verify "${REV}^{commit}")"
echo "昇格するコミット: $COMMIT  → ssh:$SSHH"; git log -1 --format='  %h %s' "$COMMIT"
if [ "$YES" != 1 ] && [ "$DRY" != 1 ]; then read -r -p "prod に反映しますか？ [y/N] " a; [ "$a" = y ] || { echo "中止"; exit 1; }; fi
# 前検査: この環境（stg）の UI ソース・フラグが、昇格するコミットと同じか（stg で確認した内容と別物を prod に出さない）
if [ "$SKIP_PARITY" != 1 ]; then PYTHONPATH="$ROOT" python3 -m src.scripts.ops.env_parity_check promote-precheck --commit "$COMMIT" || { echo "中止（承知の上なら --skip-parity）" >&2; exit 1; }; fi
[ "$DRY" = 1 ] && { echo "(dry-run) ビルド・転送・導入はしません"; exit 0; }
OUTFILE="$(bash "$ROOT/scripts/ops/vps/build_release.sh" --commit "$COMMIT" | tail -1)"
BASE="$(basename "$OUTFILE" .tar.gz)"; DIR="$(dirname "$OUTFILE")"
scp -q "$DIR/$BASE.tar.gz" "$DIR/$BASE.tar.gz.sha256" "$DIR/$BASE.manifest.json" "$SSHH:/srv/keiba-ui/incoming/"
if ssh "$SSHH" "sudo /opt/keiba-ui/bin/install_release.sh /srv/keiba-ui/incoming/$BASE.tar.gz"; then RES=ok; else RES=failed; fi
mkdir -p data/local/meta
printf '{"at":"%s","release":"%s","commit":"%s","result":"%s"}\n' "$(date -Is)" "$BASE" "$COMMIT" "$RES" >> data/local/meta/vps_promote_history.jsonl
if [ "$RES" = ok ]; then echo "prod の版:"; ssh "$SSHH" "curl -s -m 5 http://127.0.0.1:3000/version.json"; echo "  （stg との照合: python3 -m src.scripts.ops.env_parity_check compare --stg-api … --prod-api … --stg-ui …/version.json --prod-ui …/version.json）"; else echo "失敗しました（自動で直前の版に戻っています）。ログ: ssh $SSHH 'tail -30 /var/log/keiba-ui/deploy.log'" >&2; exit 1; fi
