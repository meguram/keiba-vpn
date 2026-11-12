# 別PCで Claude Code を使う

方針: 共有する指示は Git、マシン固有の設定は各PCで個別管理する。

| 種類 | 置き場所 | 別PCへの届け方 |
|---|---|---|
| プロジェクトの指示・skills・フック | `CLAUDE.md`、`.claude/settings.json`、`.claude/skills/` | `git pull` |
| ユーザー設定（モデル・サブエージェント） | `config/claude_user/` → `~/.claude/` | `bash scripts/setup_claude_user.sh` |
| 認証・許可リスト | `claude login`、`.claude/settings.local.json` | 各PCで個別（Git に入れない） |

## 手順
1. `git clone` して `develop` を取得、`bash scripts/setup_claude_user.sh`
2. `claude` を起動してログイン
3. 必要なら `.claude/settings.local.json` の許可を各PCで追加

## 会話の引き継ぎ
- `/remote-control`（元PCを起動したまま別端末から操作）
- `/teleport`（クラウドセッションを別PCへ。未コミットがあると stash を提案される）
- `/desktop`（同じPCのデスクトップアプリへ）

ユーザー設定を変えたら `config/claude_user/` にも反映して commit する。
