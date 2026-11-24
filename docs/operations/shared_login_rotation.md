# 共有ログインのパスワードの毎月の切り替え（乱数・GCP の中）

計画と経緯: `shared_login_rotation_plan_20261119.md`（2026-11-19 承認）。旧方式: `shared_login.md`。
本体: `src/scripts/ops/login_rotate.py`。Secret Manager の窓口: `src/ops/secret_store.py`。作る手順: `scripts/gcp/setup_login_rotate_job.sh`。

## 動き

```
Cloud Scheduler keiba-login-rotate（0 6 1,25-31 * *、Asia/Tokyo）
  └→ Cloud Run Job keiba-login-rotate（SA keiba-login-rotator・1 タスク・再試行 2 回）
       python -m src.scripts.ops.login_rotate run   … 日付で振り分ける
```

| 日（06:00 JST） | 役 | すること |
|---|---|---|
| 25 日〜末日の 4 日前 | 準備 | 来月の次の版（16 字の乱数）を作り、Secret Manager に置く。名簿に HMAC（`next_mac`）を書く。そろっていれば何もしない |
| 末日の 3 日前〜末日 | 下書き | 準備をやり直し、そろっていれば Slack に「下書きができた」とだけ送り、送れたら名簿に `draft_month` を書く（送れなければ翌日にやり直す）。**この後は次の版を作り直さない** |
| 1 日 | 切り替え | 下書きが済み（`draft_month`）、次の版が名簿の `next_mac` と合うときだけ、名簿のハッシュを世代の条件つきで書き換える。その後で Secret Manager を `{"current": 今月, "next": null}` に追いつかせる |

- 確定の点は名簿（GCS）。名簿が書けた後に止まっても、Job をもう一度流せば Secret Manager が追いつく。
- 名簿の `rotation_month` が今月なら切り替えない（二重の切り替えの防止）。
- 次の版が無い・名簿と合わない・下書きが済んでいないときは切り替えない。会員は先月のパスワードのまま入れる。
- 名簿を書いた後で Secret Manager の同期だけが失敗したときは、「切り替え済み・同期だけ失敗」と通知する（rollback しない。Job をもう一度流す）。
- 切り替えの前のログインは、数分以内に切れる（`pw_changed_ts`。API の名簿の取り込みは 60 秒に 1 回）。
- 切り替えの後に先月のパスワードを入れた人には「パスワードが変わりました」と案内し、失敗に数えない（名簿の `prev_mac` と比べる。`src/api/flask_auth.py`）。
  - 先月の値を確かめられなかった切り替え（切り替えの通知にその旨が出る）と rollback の後は `prev_mac` が無く、この案内は出ない（ふつうの失敗に数える。ロックは IP ごとなので影響は小さい）。
- 結果は秘密の無い `ops/login_rotate/status.json` に書く。毎朝 07:30 の `schedule-check` が次の 2 つを Slack に出す。
  - 失敗が残っているとき（今月の切り替えの失敗は、直るまで毎朝）。
  - 1 日の 07:00 以降に、その月の切り替えの成功が無いとき。

## 秘密の置き場

- `KEIBA_SHARED_LOGIN_PW`（Secret Manager）: `{"current": {"month","password"}, "next": {...} | null}`。読めるのは `keiba-login-rotator` とオーナーだけ。API・`keiba-jobs` は読まない。
- 名簿（GCS）に置くのは、ハッシュと HMAC だけ（`cur_mac`・`next_mac`・`prev_mac`・`prev`）。
- Job に渡すのは `KEIBA_MEMBERS_PEPPER`（`hex:…`。今のペッパーと同じ値）・`KEIBA_SHARED_LOGIN_ID`・`SLACK_WEBHOOK_URL`。`DEV_SECRET_KEY`（セッションの署名の鍵）は渡さない。
- Slack・ログ・status.json には、パスワード・ID・HMAC を出さない（テスト `tests/ops/test_login_rotate.py`）。

## 下書きの取り出し（オーナー・ホストで 1 行）

```
cd /home/jovyan/work/keiba-vpn && sudo gcloud secrets versions access latest --secret=KEIBA_SHARED_LOGIN_PW --project=horse-racing-466415 | sudo /opt/conda/bin/python3 -m src.scripts.ops.login_rotate draft
```

- 画面に、note に貼る文面が出る。内容は今月（as-is）と来月（to-be）のパスワード・切り替わる日時・ログインの URL・注意。ファイル・ログには残らない。
- 中身は名簿と照合してから表示する。合わなければ表示せずに止まる。
- ホストのコードが古い（pull していない）と動かない。必要なら自分用の clone で流す。
- note には 1 日 06:00 に予約投稿する（noteプレミアム）。

## はじめて動かす（ユーザー・root の gcloud。順番を守る）

1. **前提**
   - master への昇格で、このコードが jobs のイメージに入っていること（`promote.yml` の build-images）。
   - 本数の確認: `gcloud scheduler jobs list --location asia-northeast1`・`gcloud secrets list`。
   - ホストの root の gcloud が誰か: `sudo gcloud config get-value account`。
2. **ペッパーの登録**（値は画面に出ない）:
   ```
   cd /home/jovyan/work/keiba-vpn && sudo /opt/conda/bin/python3 -m src.scripts.ops.login_rotate export-pepper | sudo bash scripts/gcp/put_secret.sh KEIBA_MEMBERS_PEPPER
   ```
   - 先に 3 の SA を作る（put_secret.sh が読み出しの権限を付けるため）。順番は「3 の 1) だけ流す → 2 → 3 の全部」でよい。
3. **資源を作る**:
   - 表示だけ: `IMAGE=asia-northeast1-docker.pkg.dev/horse-racing-466415/keiba-vpn/jobs:<最新の promote の短い SHA> GCS_BUCKET=<バケット> OWNER=user:<オーナー> DEPLOYER_SA=<WIF_SERVICE_ACCOUNT の値> bash scripts/gcp/setup_login_rotate_job.sh`
   - `IMAGE` は login_rotate の入った版（`live-jobs` は ENABLE_JOBS_DEPLOY のときしか進まないので使わない）。
   - 内容を確かめてから `--apply`。Scheduler は PAUSED で作られる。
4. **試す**（本番の名簿に書くのは準備だけ。切り替えはしない）: `gcloud run jobs execute keiba-login-rotate --region asia-northeast1 --args=-m,src.scripts.ops.login_rotate,run,--step,prepare --wait`
   - 20 日以前に流すと、来月分の次の版が作られる。25 日の準備は「そろっている」で何もしない。
   - これだけでは 1 日に切り替わらない。切り替えには、末日の 3 日前以降の下書き（Slack の通知と名簿の印）が要る。
5. **旧経路を止める**:
   - `sudo bash scripts/cron/install_shared_login_cron.sh --remove`
   - 翌日、`data/local/meta/cron_runs/shared-login.tsv` に行が増えていないことを確かめる。
   - 旧経路が残ると、1 日にパスワードを派生の値へ戻す。新しいコードの `ensure` は `managed` で触らないが、ホストのツリーは pull しないので古いコードが動く。
6. **動かす**: `gcloud scheduler jobs resume keiba-login-rotate --location asia-northeast1`
7. 過去の Slack の、パスワードの入ったメッセージを消す。

## 戻し方

- **切り替えの後に戻す**: 名簿の前のハッシュへ戻す（平文を取り出さない）。
  - `gcloud run jobs execute keiba-login-rotate --region asia-northeast1 --args=-m,src.scripts.ops.login_rotate,rollback --wait`
  - 会員は先月の note のパスワードで入れる。note で知らせる。
  - 次の版は消えるので、Job がもう一度動いても切り替わらない。
  - 通知に「次の自動の切り替えは止まります」と出たとき（戻した値が Secret Manager に無い）: 翌月の下書きが「今の版が名簿と合いません」で止まる。次のどちらかで再開する。
    - 管理画面で共有の記録を消し、旧方式の `shared_login ensure` で作り直してから、準備から流す。
    - 戻した値の分かる人が、`{"current": {"month": "<今月>", "password": "<値>"}, "next": null}` を標準入力から足す（画面・履歴に出さない。例: `sudo gcloud secrets versions add KEIBA_SHARED_LOGIN_PW --data-file=- --project=horse-racing-466415` に貼って Ctrl-D）。
- **途中で止まった**: 同じ Job をもう一度流す（`--args=…,run,--step,switch`）。
- **次の版が無いまま 1 日になった**: 切り替わらない。次の順に手で流す（`--now` は前月の日付。例 `--now 2027-01-31T12:00`）。
  1. `--args=-m,src.scripts.ops.login_rotate,run,--step,draft,--now,<前月の日付>`（準備もやり直す。Slack に下書きの通知）
  2. 下書きを取り出す（取り出しの 1 行の末尾に `--now <前月の日付>` を付ける）。note を訂正する
  3. `--args=…,run,--step,switch`（今の時刻のまま）
- **新方式をやめる**: `gcloud scheduler jobs pause keiba-login-rotate` → `install_shared_login_cron.sh` で旧経路に戻す。
  - 旧経路は `rotation_month` のある記録に触らないので、その場合は管理画面で記録を作り直す。

## 費用（最悪の回数）

| 品目 | 量 | 月額 |
|---|---|---|
| Cloud Scheduler | 1 本（無料の 3 本は効いていない前提） | 15 円 |
| Secret Manager の有効な版 | パスワード 2 版（新しい 2 版を残し、名簿と合わない古い版は destroy）＋ ペッパー 1 版 | 27 円 |
| Secret Manager のアクセス | 月 24 回の実行 × 4 秘密 ≒ 100 回（1 万回まで無料） | 0 円 |
| Cloud Run Jobs | 月 24 回（8 回 × 再試行 3）× 60 秒 × 1 vCPU | 約 4 円 |
| GCS（Class A/B） | 1 回 10 件 × 24 ＋ 見張り 30 回 | 1 円未満 |
| **合計** | | **約 47 円** |

- API は Secret Manager を読まない。API の起動回数（最悪 96 回/日）の掛け算は乗らない。
- 単価は `cost_ledger.md` の 1 章（課金の実績）。
- GCP の外: noteプレミアム 月 500 円。
