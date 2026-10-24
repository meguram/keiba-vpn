#!/usr/bin/env bash
# stg の自動反映で、何を反映するかを決める。変更されたファイルから api / ui の要否を GITHUB_OUTPUT 形式（api=true|false, ui=true|false）で標準出力に出す。
#   bash scripts/ci/stg_changes.sh <変更前のコミット> <変更後のコミット>
# 迷ったら反映する側に倒す: 変更前が不明（初回・手動実行）、差分が取れない、のときは両方 true。
#   ui  : frontend/ と、画面のビルド/公開の手順（_build-ui.yml・deploy-stg.yml・scripts/ops/vps/build*）
#   api : 上記の画面・文書・テスト・ノートブックだけの変更でなければ true（コード・設定・イメージの材料はすべて対象）
set -euo pipefail
BEFORE="${1:-}"; AFTER="${2:-HEAD}"
if [ -z "$BEFORE" ] || [ "$BEFORE" = "0000000000000000000000000000000000000000" ] || ! files="$(git diff --name-only "$BEFORE" "$AFTER" 2>/dev/null)"; then
  echo "api=true"; echo "ui=true"; exit 0
fi
api=false; ui=false
while IFS= read -r f; do
  [ -n "$f" ] || continue
  case "$f" in
    frontend/*|.github/workflows/_build-ui.yml|scripts/ops/vps/build*) ui=true ;;
    .github/workflows/deploy-stg.yml) ui=true; api=true ;;
    docs/*|tests/*|notebooks/*|output/*|*.md) ;;
    *) api=true ;;
  esac
done <<< "$files"
echo "api=$api"; echo "ui=$ui"
