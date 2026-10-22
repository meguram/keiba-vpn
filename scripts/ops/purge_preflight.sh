#!/usr/bin/env bash
# 履歴除去の前の保険の確認。除去対象の『実体』が 3 か所に揃っているかを、ファイルごとの sha256 で突き合わせる（履歴除去後は git から復元できない）。
#   bash scripts/ops/purge_preflight.sh check     (a) ホストの実ファイル (b) GCS の models/fitted_artifacts/ (c) リポジトリ外のバックアップ
#   bash scripts/ops/purge_preflight.sh backup    (c) を作る: git bundle（全ブランチ）と、対象ファイルの tar.gz を BACKUP_DIR に
# 環境変数: PURGE_HOST(既定 keiba-remote) HOST_REPO(既定 ~/work/keiba-vpn) KEIBA_GCS_BUCKET(必須: (b) の確認に) BACKUP_DIR(既定 ~/keiba_private/backup)
# 読み取りのみ（backup は手元にファイルを作る）。ホスト・GCS には書かない。全部 OK で終了コード 0。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PURGE_HOST="${PURGE_HOST:-keiba-remote}"; HOST_REPO="${HOST_REPO:-~/work/keiba-vpn}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/keiba_private/backup}"
BUCKET="${KEIBA_GCS_BUCKET:-}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
bash scripts/ops/fitted_artifacts.sh list > "$TMP/files.txt"
N="$(wc -l < "$TMP/files.txt")"
xargs -d '\n' sha256sum < "$TMP/files.txt" | awk '{print $2"\t"$1}' | sort > "$TMP/local.tsv"   # path<TAB>sha

case "${1:-check}" in
  backup)
    mkdir -p "$BACKUP_DIR"; d="$(date +%Y%m%d_%H%M%S)"
    tar czf "$BACKUP_DIR/fitted_artifacts_$d.tar.gz" -T "$TMP/files.txt"
    git bundle create "$BACKUP_DIR/keiba-vpn_$d.bundle" --all >/dev/null 2>&1 && git bundle verify "$BACKUP_DIR/keiba-vpn_$d.bundle" >/dev/null 2>&1 && echo "bundle OK"
    echo "作成: $BACKUP_DIR/fitted_artifacts_$d.tar.gz ($N ファイル) と keiba-vpn_$d.bundle"; exit 0 ;;
  check) ;;
  *) echo "使い方: bash scripts/ops/purge_preflight.sh check|backup" >&2; exit 2 ;;
esac

ng=0
cmp_report() {  # 名前, 相手の path<TAB>sha ファイル
  python3 - "$TMP/local.tsv" "$2" "$1" <<'PY'
import sys
def load(p):
    d={}
    for l in open(p,encoding="utf-8"):
        l=l.rstrip("\n")
        if "\t" in l:
            a,b=l.split("\t",1); d[a]=b
    return d
loc=load(sys.argv[1]); oth=load(sys.argv[2]); name=sys.argv[3]
miss=sorted(set(loc)-set(oth)); diff=sorted(p for p in set(loc)&set(oth) if loc[p]!=oth[p])
ok=len(loc)-len(miss)-len(diff)
print(f"| {name} | {ok}/{len(loc)} | {len(miss)} | {len(diff)} |")
for p in miss[:5]: print(f"    無い: {p}", file=sys.stderr)
for p in diff[:5]: print(f"    内容が違う: {p}", file=sys.stderr)
sys.exit(0 if not miss and not diff else 1)
PY
}
echo "| 場所 | 一致 | 無い | 内容が違う |"; echo "|---|---|---|---|"

# (a) ホストの実ファイル
ssh -o BatchMode=yes -o ConnectTimeout=15 "$PURGE_HOST" "cd $HOST_REPO && xargs -d '\n' sha256sum 2>/dev/null" < "$TMP/files.txt" 2>/dev/null | awk '{print $2"\t"$1}' > "$TMP/host.tsv" || true
cmp_report "(a) ホストの実ファイル" "$TMP/host.tsv" || ng=1

# (b) GCS の models/fitted_artifacts/
if [ -z "$BUCKET" ]; then echo "| (b) GCS | 未確認（KEIBA_GCS_BUCKET が未設定） | - | - |"; ng=1
else
  G="~/.config/keiba-gcp/gcloud_ro.sh"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$PURGE_HOST" "
    v=\$($G storage cat gs://$BUCKET/models/fitted_artifacts/latest.json 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)[\"version\"])' 2>/dev/null) || exit 3
    $G storage cat gs://$BUCKET/models/fitted_artifacts/\$v/manifest.json 2>/dev/null" 2>/dev/null \
  | python3 -c 'import sys,json
try:
    m=json.load(sys.stdin)
except Exception:
    sys.exit(3)
for r in m.get("files",[]): print(r["path"]+"\t"+r["sha256"])' > "$TMP/gcs.tsv" || true
  cmp_report "(b) GCS の models/fitted_artifacts/" "$TMP/gcs.tsv" || ng=1
fi

# (c) リポジトリ外のバックアップ（最新の tar.gz と bundle）
tarf="$(ls -t "$BACKUP_DIR"/fitted_artifacts_*.tar.gz 2>/dev/null | head -1 || true)"
bund="$(ls -t "$BACKUP_DIR"/keiba-vpn_*.bundle 2>/dev/null | head -1 || true)"
if [ -n "$tarf" ]; then
  mkdir -p "$TMP/tarx"; tar xzf "$tarf" -C "$TMP/tarx" 2>/dev/null
  ( cd "$TMP/tarx" && find . -type f | sed 's#^\./##' | sort | xargs -d '\n' sha256sum 2>/dev/null | awk '{print $2"\t"$1}' ) > "$TMP/bak.tsv"
else : > "$TMP/bak.tsv"; fi
cmp_report "(c) バックアップの tar.gz" "$TMP/bak.tsv" || ng=1
if [ -n "$bund" ] && git bundle verify "$bund" >/dev/null 2>&1; then echo "| (c) バックアップの bundle | OK（$(basename "$bund")） | - | - |"; else echo "| (c) バックアップの bundle | 無い・検証失敗 | - | - |"; ng=1; fi
echo; [ $ng -eq 0 ] && echo "すべて OK: 履歴除去に進めます" || echo "NG があります: 揃うまで履歴除去をしない（ホストへ配置 / fitted_artifacts.sh pack → model_publish publish / purge_preflight.sh backup）"
exit $ng
