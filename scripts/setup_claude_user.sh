#!/usr/bin/env bash
# 別PCで Claude Code を同じ設定にする（ユーザー設定 ~/.claude/ への反映）。
# プロジェクト共有の設定（CLAUDE.md・.claude/）は git pull で届く。ここで入れるのはマシン共通のユーザー設定のみ。
# 認証（claude login）・API キー・.claude/settings.local.json は各PCで個別に用意する（Git に入れない）。
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)/config/claude_user"
DST="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
mkdir -p "$DST/agents"
cp -v "$SRC"/agents/*.md "$DST/agents/"
if [ -f "$DST/settings.json" ] && ! cmp -s "$SRC/settings.json" "$DST/settings.json"; then
  cp "$DST/settings.json" "$DST/settings.json.bak-$(date +%Y%m%d%H%M%S)"
  echo "既存の settings.json を退避しました"
fi
cp -v "$SRC/settings.json" "$DST/settings.json"
echo "完了。次に 'claude' を起動してログイン（claude login）してください。"
