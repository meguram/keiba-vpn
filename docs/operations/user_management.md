# ユーザー（ログイン ID とパスワード）の管理

| 種類 | ログイン | 管理 |
|------|----------|------|
| 管理者（開発者） | ID なし。パスワードだけ（`.env` の `DEV_PASSWORD`） | `.env` を変更して API を再起動 |
| 会員 | ログイン ID＋パスワード | `python -m src.scripts.ops.manage_users`（下記） |
| 動作確認用 `test` | `config/members.json`。`KEIBA_TEST_USER=1` のときだけ有効（公開前は off） | ここからは変更しない |

## 会員の管理（ホストで実行）

ホストでは、API と同じ conda の Python（`/opt/conda/bin/python3`）で実行する（標準の `python3` には fastapi が無く、起動できない）。

```
ssh keiba-remote 'cd /home/jovyan/work/keiba-vpn && sudo -n /opt/conda/bin/python3 -m src.scripts.ops.manage_users list'
python -m src.scripts.ops.manage_users add <id> [--name 表示名] [--generate | --password-stdin]   # 既定は対話入力（2 回）
python -m src.scripts.ops.manage_users passwd <id> [--generate | --password-stdin]
python -m src.scripts.ops.manage_users rename <id> --name 表示名
python -m src.scripts.ops.manage_users disable <id> | enable <id> | remove <id>
```

- 保存先は `data/local/auth/members.json`（**git に入れない**。環境変数 `KEIBA_MEMBERS_FILE` で変更）。パスワードはソルト付き PBKDF2-SHA256（60 万回）のハッシュだけ。ファイルの権限は 600。
- パスワードは 10 文字以上、ID と同じ文字列は不可。`--generate` で作ったパスワードは、その 1 回だけ画面に出る（記録しない）。
- API は会員ファイルの更新を自動で読み直す（再起動不要）。**無効化・削除したユーザーは、ログイン中でも次のリクエストから使えなくなる**。
- ID は英数字と `. _ @ -` の 3〜64 文字。

## 画面の右上

ログイン中は `👤 表示名`（会員は `--name`、管理者は「管理者」）、未ログインは「ゲスト」と「ログイン」ボタン。セッションの Cookie には ID が署名つきで入る（`role:時刻:ID:署名`。従来の Cookie も引き続き有効）。
