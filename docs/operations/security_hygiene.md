# 秘密・内部情報の漏洩を防ぐ運用

このリポジトリは**公開**。コミットした内容は履歴に残り、消しても取り戻せない。

## 入れてはいけないもの
- 秘密の値（Slack Webhook、トンネル／サービストークン、サービスアカウントの鍵、`.env` の値、パスワード）
- 内部情報（VPS の IP・SSH ポート・管理ユーザー名、鍵ファイルのパス、内部のホスト名の一覧）。文書・例には `<VPS_IP>` `<SSH_PORT>` `<VPS_ADMIN_USER>` のように書く。実値は gitignore 済みのローカルファイル（`scripts/ops/vps/vars.env` など）に置く。

## 検査
- `bash scripts/dev/secret_scan.sh`（push 予定の差分）／ `--tracked`（追跡中の全ファイル）。値は表示せず、行番号とパターン名だけ出す。
- push 前に自動で走らせる: clone ごとに 1 回 `git config core.hooksPath .githooks`（`.githooks/pre-push`）。
- GitHub 側（リポジトリの Settings → Code security）で **Secret scanning** と **Push protection** を有効にする（公開リポジトリは無料）。

## 値を渡すとき
- チャットや Issue に貼らない。端末で `read -rs` に入れ、ssh の標準入力で渡す。コマンド引数に値を置かない（履歴・プロセス一覧に残る）。
- 貼ってしまった値は**無効化して作り直す**（Slack Webhook は Incoming Webhooks で削除→再作成）。

## ローテーションの目安
| 種類 | きっかけ | 方法 |
|---|---|---|
| Slack Webhook | チャット・画面・ログに出た | Slack で再生成 → `.env` と Secret Manager を差し替え |
| Cloudflare トンネルトークン | 出力に出た | Zero Trust でトークンを再生成 → `/tmp/set_cf_token.sh` 相当で差し替え |
| サービスアカウント鍵 | 漏洩の疑い／不要になった | IAM → サービスアカウント → 鍵を削除（読み取り専用 `keiba-viewer` も同様） |

## ホストの後始末
- `.env` の編集前バックアップ（`.env.bak.*`）にも秘密が入っている。確認後に削除する: `sudo -n rm ~/work/keiba-vpn/.env.bak.*`
- `.env` は `root:users` の 640 のまま。

## 既知の公開済み情報（履歴に残るため取り消せない）
- 過去のコミットに VPS の IP・SSH ポート・管理ユーザー名が含まれる。現在の HEAD ではプレースホルダに置き換えた。IP を変えられる機会（VPS の作り直し）があれば変更する。
