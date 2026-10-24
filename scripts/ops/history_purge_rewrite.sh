#!/usr/bin/env bash
# 履歴除去の書き換え（リハーサルにも本番にも使う）。手元の/本物のリポジトリは変更しない: 作業用の mirror クローン上だけで書き換え、
# push はしない（push は docs/operations/history_purge_runbook.md の手順で人が行う）。
#   リハーサル: bash scripts/ops/history_purge_rewrite.sh [作業ディレクトリ]            既定 /tmp/purge_rehearsal（手元のリポジトリから mirror）
#   本番:       CLONE_FROM=https://github.com/meguram/keiba-vpn.git bash scripts/ops/history_purge_rewrite.sh /tmp/purge_real
# 手順: mirror クローン → git filter-branch（config/history_purge_paths.txt のパスを全履歴から削除）→
#       develop の先端に、config/history_purge_restore.txt の『書き換え前の HEAD の版』を戻す 1 コミットを追加 → 検証。
set -euo pipefail
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${1:-/tmp/purge_rehearsal}"
BR="${BR:-develop}"
rm -rf "$WORK"; mkdir -p "$WORK"
git clone -q --mirror "${CLONE_FROM:-$SRC}" "$WORK/mirror.git"
cd "$WORK/mirror.git"
# ローカル由来の remote-tracking ref は対象外（本番手順では GitHub から直接 mirror クローンするので存在しない）
git for-each-ref refs/remotes --format='%(refname)' | xargs -r -n1 git update-ref -d
echo "対象ブランチ: $(git for-each-ref refs/heads --format='%(refname:short)' | tr '\n' ' ')"
ORIG="$(git rev-parse "refs/heads/$BR")"
git ls-tree -r "$ORIG" > "$WORK/tree_before.txt"
git for-each-ref refs/heads --format='%(refname:short) %(objectname)' > "$WORK/heads_before.txt"
echo "元の $BR: $ORIG  サイズ(.git): $(du -sh . | cut -f1)"

# pathspec 配列を作る（glob: は :(glob) に、末尾 / はディレクトリ）
python3 - "$SRC/config/history_purge_paths.txt" > "$WORK/pathspecs.txt" <<'PY'
import sys
for l in open(sys.argv[1], encoding="utf-8"):
    l = l.strip()
    if not l or l.startswith("#"): continue
    print((":(glob)" + l[5:]) if l.startswith("glob:") else l)
PY
echo "除去パス数: $(wc -l < "$WORK/pathspecs.txt")"

export FILTER_BRANCH_SQUELCH_WARNING=1
git filter-branch -f --prune-empty --index-filter \
  "xargs -d '\n' -a '$WORK/pathspecs.txt' git rm -r --cached --ignore-unmatch -q --" -- --all 2>&1 | tail -3

# 復元コミット（develop のみ）: 書き換え前の先端から、整理済みの版を戻す
NEWTIP="$(git rev-parse "refs/heads/$BR")"
export GIT_INDEX_FILE="$WORK/idx_restore"
git read-tree "$NEWTIP"
while IFS= read -r p; do
  case "$p" in ''|'#'*) continue;; esac
  line="$(git ls-tree "$ORIG" -- "$p" | head -1)"
  [ -z "$line" ] && continue
  mode="${line%% *}"; rest="${line#* }"; sha="${rest#* }"; sha="${sha%%$'\t'*}"
  git update-index --add --cacheinfo "$mode,$sha,$p"
done < "$SRC/config/history_purge_restore.txt"
TREE="$(git write-tree)"; unset GIT_INDEX_FILE
RESTORE="$(git commit-tree "$TREE" -p "$NEWTIP" -m '履歴整理後の復元: 整理済みの版を戻す（ノートブック・カタログなど）')"
git update-ref "refs/heads/$BR" "$RESTORE"

# 後始末（バックアップ ref を消して、残存を正しく検証できるようにする）
rm -rf refs/original; git reflog expire --expire=now --all; git gc -q --prune=now --aggressive

# 検証
echo "== 検証1: 除去対象が全履歴に残っていないか"
git log --all --name-only --format= > "$WORK/paths_after.txt"
python3 - "$WORK/pathspecs.txt" "$WORK/paths_after.txt" "$SRC/config/history_purge_restore.txt" <<'PY'
import sys, fnmatch
specs=[l.rstrip("\n") for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
after=sorted({l.rstrip("\n") for l in open(sys.argv[2], encoding="utf-8") if l.strip()})
bad=[]
for s in specs:
    if s.startswith(":(glob)"): m=[p for p in after if fnmatch.fnmatch(p, s[7:])]
    elif s.endswith("/"): m=[p for p in after if p.startswith(s)]
    else: m=[p for p in after if p==s]
    bad+=m
# 復元したパスは履歴に（復元コミットで）現れるので除外して確認
restore={l.strip() for l in open(sys.argv[3], encoding="utf-8") if l.strip() and not l.startswith("#")}
bad=[p for p in bad if p not in restore]
print("残っている除去対象:", len(bad)); [print(" ",p) for p in bad[:10]]
PY
echo "== 検証2: 先端のツリー差分（元の先端 → 書き換え後の先端）"
git ls-tree -r "$(git rev-parse refs/heads/$BR)" > "$WORK/tree_after.txt"
python3 - "$WORK/tree_before.txt" "$WORK/tree_after.txt" <<'PY'
import sys
def load(p):
    d={}
    for l in open(p,encoding="utf-8"):
        meta,path=l.rstrip("\n").split("\t",1); d[path]=meta.split()[2]
    return d
a=load(sys.argv[1]); b=load(sys.argv[2])
gone=sorted(set(a)-set(b)); new=sorted(set(b)-set(a)); chg=sorted(p for p in set(a)&set(b) if a[p]!=b[p])
print(f"先端: 前 {len(a)} ファイル → 後 {len(b)} ファイル / 消えた {len(gone)} / 増えた {len(new)} / 内容が変わった {len(chg)}")
for t,l in (("消えた",gone),("増えた",new),("内容が変わった",chg)):
    for p in l[:8]: print(f"  {t}: {p}")
PY
echo "== ブランチ（書き換え後）"
git for-each-ref refs/heads --format='%(refname:short)' | tr '\n' ' '; echo
echo "== 検証3: サイズ"
echo "書き換え後(.git 相当): $(du -sh . | cut -f1)"
git rev-list --all | wc -l | sed 's/^/コミット数(書き換え後): /'
echo "書き換え完了（push はしていない）。mirror: $WORK/mirror.git / 書き換え前の各ブランチ: $WORK/heads_before.txt"
