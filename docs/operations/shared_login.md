# 共通ログイン（特定の少人数に配る 1 つの ID）— 毎月 1 日に自動更新

## 動き
- 毎日 **00:05 JST** に cron（`scripts/cron/shared_login_daily.sh`）が `python -m src.scripts.ops.shared_login ensure` を実行する。月が替わっていれば、共通 ID のパスワードを新しい月のものに更新する（それ以外の日は何もしない）。
- 更新＝パスワード変更なので、**それ以前のログイン（cookie・自動ログイン）はすべて使えなくなる**（`pw_changed_ts`。`src/api/auth.py`）。
- 新しいログイン情報（ID・パスワード）は **Slack**（`.env` の `SLACK_WEBHOOK_URL`）に通知する。送れなかったときは翌日の実行で再送する。
- 月ごとのパスワードは「サーバの秘密（ペッパー）+ 年月」から決まる値で、**ファイルには保存しない**。いつでも `python -m src.scripts.ops.shared_login show` で今月分を表示できる（ホストの root）。
- ID は `.env` の `KEIBA_SHARED_LOGIN_ID`（リポジトリには書かない）。未設定なら何もしない。管理画面でそのユーザーを「無効」にすると、cron は触らない。

## 2026-10-23〜11-13 の時刻のずれ
- 登録スクリプト（`install_shared_login_cron.sh`）が、JST の `5 0 * * *` をそのまま UTC の crontab に書いていた。そのため、実際は毎日 **09:05 JST** に動いていた（`data/local/meta/cron_runs/shared-login.tsv` で確認）。
- 影響:
  - 毎月 1 日の 00:00〜09:05 は、前月のパスワードが通ったままだった。
  - 09:05 の更新で、それまでのログインがすべて切れた。2026-11-01（日・開催日）は朝 09:05 に締め出しが起きた。
  - 新しいパスワードの Slack 通知も 09:05 だった。
- 2026-11-13 に登録スクリプトを `5 15 * * *`（UTC＝JST 00:05）に直した（7c4a8adb）。**root の crontab の入れ直しは未実施**（権限の確認で止まった。`docs/operations/handoff_20261113_pending.md` の ④、ユーザーの作業待ち）。入れ直すまでは毎日 09:05 のまま（次の月替わりは 12/01 の 09:05）。毎日の実行は、月が替わっていなければ何もしない（冪等）。

## 設定
1. ホストの `.env` に `KEIBA_SHARED_LOGIN_ID=<ID>` を追加（`SLACK_WEBHOOK_URL` は既存）。
2. cron 登録: `sudo bash scripts/cron/install_shared_login_cron.sh`（冪等。外すのは `--remove`）。
3. 初回は `ensure` を手で 1 回実行すると、アカウントが作られて Slack に通知される。

## 注意
- Slack に残るパスワードは、配布後にメッセージを削除する。
- 秘密（`DEV_SECRET_KEY` / `KEIBA_MEMBERS_PEPPER`）を変えると、月ごとのパスワードも変わる（次回の `ensure` で更新・通知される）。
- 個別に止めたい人が出てきたら、共通 ID をやめて人ごとの ID（`manage_users`）に切り替える。
テスト: `python3 -m pytest tests/ops/test_shared_login.py`
