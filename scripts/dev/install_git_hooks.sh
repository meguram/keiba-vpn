#!/usr/bin/env bash
# リポジトリで管理しているフック（scripts/dev/git-hooks/）を .git/hooks/ に写す。
# worktree から動かしても、フックの置き場は共有の .git/hooks（git rev-parse --git-common-dir）。
set -euo pipefail
SRC="$(cd "$(dirname "$0")/git-hooks" && pwd)"
HOOKS="$(cd "$(git rev-parse --git-common-dir)" && pwd)/hooks"
mkdir -p "$HOOKS"
for f in "$SRC"/*; do
  name="$(basename "$f")"
  install -m 755 "$f" "$HOOKS/$name"
  echo "入れた: $HOOKS/$name"
done
