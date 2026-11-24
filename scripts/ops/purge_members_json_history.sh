#!/usr/bin/env bash
# config/members.json を GitHub の全ブランチ・全タグの履歴から除く（段 2。ユーザーが自分で流す。手順: docs/operations/members_json_purge_20261119.md）
# 既定は「写しを取って書き換え、確かめる」まで。force push は --push を付け、確認の文字を打ったときだけ行う。
#   bash scripts/ops/purge_members_json_history.sh <作業場所>          # 書き換えと確認まで（GitHub は変えない）
#   bash scripts/ops/purge_members_json_history.sh <作業場所> --push   # 写しから取り直して書き換え・確認し、force push
# 前提: git-filter-repo（docs の手順 1 で git の版と合わせて入れる）・gh の認証。作業場所はどの git の管理下でもない場所。
set -euo pipefail

REPO="meguram/keiba-vpn"
REPO_URL="https://github.com/${REPO}.git"
TARGET="config/members.json"
[ $# -ge 1 ] && [ "${1#-}" = "$1" ] || { echo "使い方: $0 <作業場所> [--push]" >&2; exit 2; }
WORK="$(realpath -m "$1")"
PUSH=0; [ "${2:-}" = "--push" ] && PUSH=1
command -v git-filter-repo >/dev/null || { echo "git-filter-repo がありません（docs の手順 1）" >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"   # このスクリプトのあるリポジトリ
case "$WORK/" in "$ROOT"/*) echo "作業場所はリポジトリの外にしてください: $WORK" >&2; exit 2 ;; esac
mkdir -p "$WORK"; chmod 700 "$WORK"
if git -C "$WORK" rev-parse --git-dir >/dev/null 2>&1; then echo "作業場所が git の管理下です: $WORK" >&2; exit 2; fi
GITC=(git -c credential.helper= -c "credential.helper=!gh auth git-credential")
heads() { git for-each-ref --format='%(objectname) %(refname)' refs/heads refs/tags | sort; }
remote_heads() { "${GITC[@]}" ls-remote "$REPO_URL" 'refs/heads/*' 'refs/tags/*' | awk '{print $1" "$2}' | grep -v '\^{}$' | sort; }

cd "$WORK"
# 1. 写し（書き換える前の丸ごとの控え。戻すときに使う）。--push のときは、その時点の GitHub から取り直す
if [ "$PUSH" = 1 ] || [ ! -d backup.git ]; then
  rm -rf backup.git
  "${GITC[@]}" clone -q --mirror "$REPO_URL" backup.git
  tar czf "backup-$(date +%Y%m%d%H%M%S).tar.gz" backup.git
fi
blobs="$(git -C backup.git rev-list --all --objects -- "$TARGET" | awk -v t="$TARGET" '$2==t{print $1}' | sort -u)"
echo "消す対象の版: $(echo "$blobs" | grep -c . ) 件"

# 2. 書き換える写し（backup.git から作る＝控えと同じ内容を書き換える）
rm -rf rewrite.git
git clone -q --mirror backup.git rewrite.git
cd rewrite.git
git for-each-ref --format='%(refname)' refs/pull refs/remotes | xargs -r -n1 git update-ref -d   # PR の参照は GitHub 側で書けないので落とす
heads > "$WORK/before_heads.txt"
git filter-repo --force --invert-paths --path "$TARGET"
mkdir -p "$WORK/filter-repo-maps"; cp -f .git/filter-repo/commit-map .git/filter-repo/ref-map "$WORK/filter-repo-maps/" 2>/dev/null \
  || cp -f filter-repo/commit-map filter-repo/ref-map "$WORK/filter-repo-maps/"   # GitHub サポートへの依頼に使う

# 3. 確かめる（どれか 1 つでも残っていたら止める）
left="$(git log --all --oneline -- "$TARGET" | wc -l)"
objs="$(git rev-list --all --objects | awk -v t="$TARGET" '$2==t' | wc -l)"
same=0; for b in $blobs; do git cat-file -e "$b" 2>/dev/null && same=$((same+1)); done   # 別の名前で残っていないか（中身で）
echo "残った commit: $left / 残ったパス: $objs / 残った中身: $same"
[ "$left" = 0 ] && [ "$objs" = 0 ] && [ "$same" = 0 ] || { echo "履歴にまだ残っています。push しません" >&2; exit 3; }
heads > "$WORK/new_heads.txt"
echo "書き換え前: $WORK/before_heads.txt / 後: $WORK/new_heads.txt / 対応表: $WORK/filter-repo-maps/"

[ "$PUSH" = 1 ] || { echo "書き換えと確認まで終わりました。push は --push を付けて流し直してください"; exit 0; }

# 4. force push の前の確かめ
echo "GitHub の全ブランチを書き換えます。続けるなら PURGE と打ってください"; read -r ans; [ "$ans" = PURGE ] || { echo "やめました"; exit 1; }
for w in promote.yml deploy-stg.yml ci.yml; do gh workflow disable "$w" -R "$REPO"; done   # 失敗したら set -e で止まる
for w in promote.yml deploy-stg.yml ci.yml; do
  st="$(gh workflow view "$w" -R "$REPO" --json state --jq .state)"
  [ "$st" = disabled_manually ] || { echo "$w が止まっていません（$st）。push しません" >&2; exit 4; }
done
busy="$(gh run list -R "$REPO" --limit 50 --json status --jq '[.[]|select(.status=="queued" or .status=="in_progress" or .status=="waiting")]|length')"
[ "$busy" = 0 ] || { echo "実行中・待ちの run が $busy 件あります。終わってから流し直してください" >&2; exit 4; }
diff <(remote_heads) "$WORK/before_heads.txt" >/dev/null || { echo "写しの後に GitHub が更新されています。--push で流し直してください" >&2; exit 4; }

# 5. force push（取った写しの時点から変わっていないブランチだけを上書きする）
lease=(); while read -r sha ref; do lease+=("--force-with-lease=$ref:$sha"); done < "$WORK/before_heads.txt"
"${GITC[@]}" push "${lease[@]}" --force "$REPO_URL" 'refs/heads/*:refs/heads/*'
if git for-each-ref refs/tags | grep -q .; then "${GITC[@]}" push "${lease[@]}" --force "$REPO_URL" 'refs/tags/*:refs/tags/*'; fi
if diff <(remote_heads) "$WORK/new_heads.txt"; then echo "GitHub の先頭が書き換え後と一致しました"; else echo "GitHub の先頭が書き換え後と一致しません（上の差を確かめる）" >&2; exit 5; fi
echo "各作業ツリーの付け直し（docs の 5.）が済んでから、workflow を戻してください:"
echo "  for w in ci.yml deploy-stg.yml promote.yml; do gh workflow enable \$w -R $REPO; done"
