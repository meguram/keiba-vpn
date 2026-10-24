#!/usr/bin/env bash
# .github/workflows/*.yml の `uses: owner/repo@tag   # PIN` を、コミット SHA 固定（`@<40桁> # tag`）に書き換える。
# GitHub の公開リポジトリに git ls-remote で問い合わせる（読み取りのみ）。実行はユーザーの端末（または私が許可を得て）。
#   bash scripts/ci/pin_actions.sh            # 解決結果を表示するだけ
#   bash scripts/ci/pin_actions.sh --write    # ファイルを書き換える
# 注釈付きタグは ^{} の指すコミットを使う。解決できなかったものは、そのまま残して警告する。
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
WRITE=0; [ "${1:-}" = "--write" ] && WRITE=1
declare -A SEEN
FOUND=0
for F in .github/workflows/*.yml; do
mapfile -t LINES < <(grep -nE '^\s*-?\s*uses:\s*[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(/[A-Za-z0-9_./-]+)?@[A-Za-z0-9_.-]+\s*#\s*PIN\s*$' "$F" || true)
[ "${#LINES[@]}" -gt 0 ] || continue
FOUND=1
for l in "${LINES[@]}"; do
  ln="${l%%:*}"; body="${l#*:}"
  ref="$(echo "$body" | sed -E 's/.*uses:\s*([^ ]+)\s*#.*/\1/')"; repo="${ref%%@*}"; tag="${ref##*@}"
  owner_repo="$(echo "$repo" | cut -d/ -f1-2)"
  key="$owner_repo@$tag"
  if [ -z "${SEEN[$key]:-}" ]; then
    sha="$(git ls-remote "https://github.com/$owner_repo" "refs/tags/$tag^{}" "refs/tags/$tag" | awk 'NR==1{print $1}')"
    SEEN[$key]="${sha:-}"
  fi
  sha="${SEEN[$key]}"
  if [ -z "$sha" ]; then echo "警告: $key を解決できません（$F 行 $ln）" >&2; continue; fi
  echo "$repo@$tag -> $sha"
  if [ "$WRITE" = 1 ]; then
    sed -i -E "${ln}s|uses:[[:space:]]*[^ ]+[[:space:]]*#[[:space:]]*PIN[[:space:]]*\$|uses: $repo@$sha   # $tag|" "$F"
  fi
done
done
[ "$FOUND" = 1 ] || { echo "固定対象（# PIN 付き）はありません"; exit 0; }
[ "$WRITE" = 1 ] && echo "書き換えました: .github/workflows/*.yml" || echo "（--write で書き換え）"
