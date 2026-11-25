# 共有アカウントのパスワードの毎月の切り替え — 基盤の計画（2026-11-19・承認済み）

担当: member-rotation。2026-11-19 にユーザーが承認。実装と運用の手順は `shared_login_rotation.md`（コマンドは `python -m src.scripts.ops.login_rotate`。この計画の `shared_login run` は、そちらに置き換えた）。
関係: `shared_login.md`（今の仕組み）・`members_store_gcs.md`（名簿の正本）・members-purge（`config/members.json` を git から外す。重ねない）。
レビュー: 別エージェントの指摘 15 件（【重】5・【中】6・【軽】4）を反映した（9 章）。
実装の後の見直し（critic・2026-11-19）: ペッパーの秘密の版 1 つ（約 9 円）を足し、費用は最悪 約 47 円/月。切り替えは下書きが済んだ月だけにした。詳しくは `shared_login_rotation.md`。

## 0. 結論

- 毎月 1 日 06:00 JST の切り替えを、GCP の中で動かす。Cloud Scheduler 1 本から、専用の Cloud Run Job 1 つを起こす。ホストが止まっていても切り替わる。
- パスワードは、今の「ペッパー＋年月から決まる値」をやめ、安全な乱数（`secrets`）で作る。平文の置き場は Secret Manager の 1 つの秘密だけ。
- 確定の点は名簿（GCS・世代の条件つきの書き込み）。Secret Manager は名簿に追いつかせる。2 か所がずれたら、再実行でそろえる。
- 費用は最悪で月 約 36 円（Scheduler 15・Secret Manager の版 18・Jobs 約 2・GCS 1 未満）。API（Cloud Run）は Secret Manager を読まない。
- Slack に平文のパスワードを送るのをやめる。通知は「成功・失敗・下書きができた」だけにする。
- **新方式の開始は 2027-01-01 を推奨**（準備 12/25・下書き 12/28）。12/01 に間に合わせるには、11/24 までに master への昇格（ユーザーの指示）と GCP の作業が要る（6 章）。12 月は今の経路のまま。

## 1. 棚卸し（今の毎月の変更）

| 項目 | 今 | 問題 |
|---|---|---|
| 動く場所 | ホストの root の cron（`KEIBA-SHARED-LOGIN`。`shared_login_daily.sh` → `shared_login ensure`）。毎日起動し、月が替わっていれば更新する | 登録スクリプトは 00:05 JST に直したが、crontab の入れ直しが未実施で、実際は 09:05 JST に動く（`shared_login.md`）。ホストが止まると切り替わらない |
| GCP の Job | `config/gcp_job_map.json` の `shared-login` は `KEIBA_SHARED_LOGIN_DRY=1`（判定だけ）。起こす Scheduler（`keiba-g-hourly`）は 11/13 に費用のため削除した | 動いていない |
| 平文の作り方 | `password_for(月)` = HMAC(ペッパー, "shared-login:"+年月) を 31 文字種の 16 字にしたもの。保存しない | ペッパー（`DEV_SECRET_KEY` から導く）が漏れると、過去・未来の全月のパスワードが分かる |
| 平文の置き場 | ファイル・GCS には無い。`show` で計算して出す。**Slack に平文で送っている**（`message()`） | Slack の履歴に残る（こちらからは確かめられない） |
| ログ | `logs/shared_login.log` は `ok`・`rotated`・`created` の 27 行だけ。パスワードの形（4-4-4-4）は 0 件（件数だけで確認） | なし |
| git | 追跡中の名簿らしいファイルは `config/members.json`（試験用 test の 1 件）だけ。members-purge が外す（82344ef0・push 待ち） | push 待ち |
| ハッシュの方式 | PBKDF2-SHA256・60 万回・塩 16 バイトの結果に、ペッパーで HMAC をかける（`algo=pbkdf2_sha256_pepper`。`src/api/auth.py:360`）。ID も HMAC（idk）で、平文を持たない | 方式は十分（OWASP の PBKDF2-SHA256 の目安は 60 万回）。変えない |
| 名簿の正本 | GCS。ホストの `.env` と GitHub の Variables の両方で `MEMBERS_STORE=gcs` を確認。書き込みは `manage_users._transact`（世代の条件つき） | なし |
| 先月の人への案内 | `flask_auth._rotated_shared_password` が `password_for(先月)` を計算して比べる。共有の ID かどうかも「今月の値がハッシュと合うか」で見分ける | 乱数にすると計算できない。作り直す（3.4） |
| 試験用アカウント | 本番の配備は `API_TEST_USER=0`（`scripts/ci/deploy_gcp.sh:137`。stg のタグつきの版だけ 1）。本番に `KEIBA_TEST_USER` があれば配備を止める検査もある（`deploy_cloud_run_api.sh:159`）。test の記録は members-purge で git から消える | 守りは既にある。足さない |
| 署名の鍵とペッパー | ペッパーは `DEV_SECRET_KEY` から導く（`KEIBA_MEMBERS_PEPPER` は未設定）。`DEV_SECRET_KEY` はセッションの署名にも使う | Job に `DEV_SECRET_KEY` を渡すと、セッションを偽造できる権限になる。ペッパーを分ける（3.5） |

## 2. 形（推奨）

```
Cloud Scheduler keiba-login-rotate（0 6 1,25-31 * *、Asia/Tokyo。PAUSED で作り、旧経路を止めた後に再開）
  └→ Cloud Run Job keiba-login-rotate（専用の SA keiba-login-rotator・タスク 1・max-retries 2）
       python -m src.scripts.ops.shared_login run   … 日付で振り分ける
         ・25 日〜: 来月の「次の版」が無ければ作る（準備。下書きの後は作らない）
         ・末日の 3 日前: 次の版の整合を確かめ、「下書きができた」と通知する（下書き）
         ・1 日: 次の版を有効にする（切り替え）
```

- **3 つの役を 1 本の Scheduler にまとめる**。Cloud Scheduler の cron（unix-cron）には月末の指定（`L`）が無いため、25〜31 日と 1 日に起こして中で判断する。1 か月の起動は最大 8 回。
  - 下書きの日は「末日の 3 日前」: 11 月なら 27 日、12 月なら 28 日、2 月なら 25 日（閏年は 26 日）。2 月は準備と同じ日になるが、同じ起動の中で準備 → 下書きの順に流す。
- **名前は `keiba-g-` で始めない**。`src/ops/cost_guard.py:38-39` が `keiba-g-` の Scheduler を止めるため（月初めに黙って止まるのを防ぐ）。台帳（`cost_ledger.md`）には載せる。
- **専用の Job・SA にする**。job_dispatch（`keiba-jobs`）に載せると、パスワードの秘密とペッパーを、多くのジョブが使う SA に渡すことになる。並走の包みは `parallel_guard: false` で外せるので、そこは理由にしない。
- ホストの cron で済ませる案は採らない。理由は 2 つ。ホストが止まると切り替わらない（実際に止まったことがある）。root の crontab の入れ直しが要る。費用の差は月 15〜30 円。

## 3. 作るもの

### 3.1 秘密の置き場（Secret Manager）

- 秘密 `KEIBA_SHARED_LOGIN_PW` を 1 つ置く。中身は JSON: `{"current": {"month": "2027-01", "password": "…"}, "next": {"month": "2027-02", "password": "…"} | null}`。
- 書き換えるたびに版を足す。古い版は、名簿の確定の後に「名簿の `current`・`next` と合わない版」だけを destroy する（順番では消さない。二重に起動したときに、相手が頼っている版を消さないため）。有効な版は、ふだん 1〜2。
- 「今の版」「次の版」は JSON の `current`・`next` で表す。版のエイリアスは使わない（エイリアスの更新には秘密そのものの更新権限が要り、既定のロールでは広すぎる）。
- 読めるのは `keiba-login-rotator` とオーナーの人のアカウントだけ。API・`keiba-jobs` には渡さない。
  - ホストの root の gcloud がどのアカウントで認証しているか（人か、SA の `megu-keiba@` か）は、7 章の手順の 0 で確かめる。SA なら、その SA には読ませず、オーナーの人のアカウントにだけ読み出しを付ける。

### 3.2 名簿の欄（共有アカウントの記録にだけ足す。どれもハッシュか HMAC）

| 欄 | 中身 | 使い道 |
|---|---|---|
| `rotation_month` | 今の版の月（例 `2027-01`） | 二重の切り替えを防ぐ。同じ月なら何もしない |
| `next_month`・`next_mac` | 次の版の月と HMAC(ペッパー, "shared-next:"+平文) | Secret Manager の次の版が、準備したものと同じかを確かめる |
| `draft_month` | 下書きを通知した月 | これがあれば、その月の次の版は作り直さない |
| `prev_mac` | HMAC(ペッパー, "shared-prev:"+先月の平文) | 先月の人への案内（3.4） |
| `prev_salt`・`prev_hash`・`prev_month` | 切り替え前のハッシュ（1 か月だけ） | 戻し（5 章）。平文を取り出さずに戻せる |

- 平文は 16 字・31 文字種（約 79 ビット）の乱数なので、HMAC のような速いハッシュでも総当たりは成り立たない（名簿とペッパーが両方漏れても）。

### 3.3 3 つの役（`shared_login run`）

共通:
- 名簿は GCS から世代つきで読む（`manage_users._current()`）。手元のファイル（`auth.member_record`）は使わない。専用の Job の新しいコンテナには手元の名簿が無いため。
- `gcs_store_enabled()` が偽なら、何もせず失敗で終わる（コンテナの中のファイルに書いて成功を返す事故を防ぐ）。
- 共有の記録は、ID（`KEIBA_SHARED_LOGIN_ID`）の idk で探す。**見つからなければ作らずに止めて通知する**。idk はペッパーつきなので、見つかることが「Job のペッパーが API と同じ」ことの確かめになる。
- 判定（`rotation_month`・`next_mac`・`draft_month`）は、`_transact` の変更の関数の中でやり直す。世代の衝突で読み直したときにも効くようにするため。
- 終わったら、結果の小さなファイル `ops/login_rotate/status.json`（月・役・結果・時刻。秘密は無し）を GCS に書く（3.6 の見張りが読む）。

**準備（25 日〜。下書きの前なら、毎日やり直してよい）**
1. 名簿に来月の `next_mac` があり、Secret Manager の `next` がそれと合えば、何もしない。
2. 名簿の `draft_month` が来月なら、作り直さない。合わないときは止めて、失敗を通知する（note に出した値と切り替わる値がずれるのを防ぐ）。
3. 無ければ、`secrets.token_bytes` から 16 字を作り、`next` を足した版を Secret Manager に書く。続けて、名簿に `next_month`・`next_mac` を世代の条件つきで書く。
   - 名簿の側に先に別の `next_mac` があれば（同時の実行）、名簿の値を正とし、自分の版を destroy する。

**下書き（末日の 3 日前の 06:00。失敗したら翌日の起動でやり直す）**
1. 準備を流す（抜けていれば作る）。
2. Secret Manager の `current`・`next` と、名簿の `rotation_month`・`next_mac` がそろっていることを確かめる。
3. 名簿に `draft_month` を書き、Slack に「下書きができました（1 月分）。取り出しは docs の 1 行」とだけ送る。
- 文面は保存しない。オーナーがホストで次の 1 行を流し、その場で作る（平文の写しを増やさない。保存するなら秘密がもう 1 つ要り、月 約 9 円かかる）。

```
cd /home/jovyan/work/keiba-vpn && sudo gcloud secrets versions access latest --secret=KEIBA_SHARED_LOGIN_PW --project=horse-racing-466415 | sudo /opt/conda/bin/python3 -m src.scripts.ops.shared_login draft
```

- `draft` は、標準入力の JSON をそのままは信じない。ホストのペッパー（`.env`）で HMAC を作り、名簿の `next_mac`・今のハッシュと合うことを確かめてから表示する。合わなければ表示せずに止める。
- 表示する中身: 今月（as-is）と来月（to-be）のパスワード・切り替わる日時（1 日 06:00 JST。数分以内に全員がログインし直しになる）・ログインの URL・注意。画面に出すだけで、ファイル・ログ・ネットワークには出さない。

**切り替え（1 日 06:00）**
1. 名簿の `rotation_month` が今月なら、Secret Manager を追いつかせて終わる（二重の切り替えの防止・途中で止まったときの再開）。
2. Secret Manager の `next` が今月で、その HMAC が名簿の `next_mac` と合うことを確かめる。**無い・合わないときは切り替えず、失敗を通知する**（誰も知らないパスワードに変えない。会員は先月のパスワードのまま使える）。
3. 名簿に、1 回の `_transact` で世代の条件つきで書く。ここが確定の点。
   - 新しい `salt`・`hash`・`pw_changed_ts`
   - `rotation_month`・`prev_*`・`prev_mac`
   - `next_*`・`draft_month` を消す
4. GCS から名簿を読み直し、新しい平文で照合できることを確かめる。
5. Secret Manager に `{"current": 今月, "next": null}` の版を足し、合わない版を destroy する。
6. Slack に「切り替えました（1 月分）」とだけ送る。失敗のときは、どの段で止まったかだけを送る。
- 旧いログインは、既存の仕組み（`pw_changed_ts`）で切れる。温まった API のインスタンスは、名簿の取り込み（要求をきっかけに 60 秒に 1 回。`gcs_mirror.py`）の後に切れるので、06:00 から数分以内になる。

### 3.4 先月の人への案内の作り直し

- `flask_auth._rotated_shared_password` を、記録の `prev_mac` と HMAC を比べる形にする。`prev_mac` を持つ記録が共有の記録になる（ほかの会員・存在しない ID では偽）。
- ID の有無で計算量を変えない（存在しない ID でも同じ HMAC を 1 回計算する）。PBKDF2 は回さないので、今より軽い。
- 初回の切り替えでは、先月の平文を `password_for(先月)`（今の方式の値）で作る。2 回目からは Secret Manager の `current` を使う。
- `password_for`・`is_previous_password` は初回の後に消す。テスト（`tests/api/test_login_latency.py`・`tests/ops/test_shared_login.py`）も直す。

### 3.5 ペッパーを分ける（Job に `DEV_SECRET_KEY` を渡さない）

- 今のペッパー（`DEV_SECRET_KEY` から導いた値）を、`KEIBA_MEMBERS_PEPPER` として Secret Manager に登録する。値を求めて登録するのはユーザー（ホスト）が行う（`login_lockout_and_key_rotation_plan.md` の 2 章と同じ手順）。
- API とホストの `.env` も、同じ値の `KEIBA_MEMBERS_PEPPER` を読むようにする。同じ値なので、今の会員の照合は変わらない。
- Job には `KEIBA_MEMBERS_PEPPER` だけを渡す。値が食い違えば、共有の記録が idk で見つからず止まる（3.3）。全員を締め出す前に気づける。

### 3.6 旧経路を止める（順番: 旧を止める → 確かめる → 新を動かす）

1. コード: `shared_login ensure` は、記録に `rotation_month` があれば何も書かない。Slack への平文の通知（`message()`）と `show` を消す。
   - ただし、ホストの共有ツリーは pull しない運用なので、この守りはホストに届かない。だから 2 の crontab の外しが必須。
2. ユーザー: root の crontab から `KEIBA-SHARED-LOGIN` の行を外す（`sudo bash scripts/cron/install_shared_login_cron.sh --remove`）。翌日、`data/local/meta/cron_runs/shared-login.tsv` に行が増えていないことを確かめる。
3. `config/gcp_job_map.json` の `shared-login` を `enabled: false` にする。
4. 2 を確かめた後で、Scheduler を再開する（`gcloud scheduler jobs resume keiba-login-rotate`）。
5. ユーザー: 過去の Slack のパスワードのメッセージを消す。

### 3.7 守り

- **CI の検査**: `scripts/dev/secret_scan.sh` に足す。CI は `--tracked` で全ファイルを見ている（`_test.yml:63`）。`data/local/` は既に検出している。
  - ファイル名: `members*.json`・`shared_login_state.json`
  - 中身: `$2a$`・`$2b$`・`$2y$`（bcrypt）、`$argon2…$`、`"salt": "<16 進 32 字>"`、`"hash"`・`"idk"`: `"<16 進 64 字>"`
  - 今これに当たるのは `config/members.json` だけ（件数で確認）。**members-purge が develop に入った後に入れる**（先に入れると CI が落ちる）。
- **ログ・例外**: 平文・HMAC・Secret Manager の中身を出さない。例外の文は段の名前だけにする。`run` の出力・Slack の文面にパスワードが含まれないことを、テストで確かめる。
- **見張り**: Job が通知の前に落ちたときのため。毎朝 07:30 の `schedule-check`（`keiba-g-0730-shadow`、既にある）に、「1 日なのに `ops/login_rotate/status.json` が今月の切り替えの成功でない → Slack」を足す。
  - 読むのは秘密の無い小さなファイル 1 つ。`keiba-jobs` に秘密を渡さない。費用は 0 円（東京の中で Class B を 1 日 1 回）。
- **GCS の権限を絞る**（必須）: rotator には IAM の条件つきで `objectUser` を付ける。対象は `ops/mirror/data/local/auth/members.json`・`ops/mirror/_index/auth.`・`ops/login_rotate/` で始まる名前だけ。

## 4. 費用（最悪の回数）

| 品目 | 量（最悪） | 単価（`cost_ledger.md` 1 章） | 月額 |
|---|---|---|---|
| Cloud Scheduler | 1 本（無料の 3 本は効いていない前提。台帳の実績どおり） | 約 15 円/本 | 15 円 |
| Secret Manager の有効な版 | 2 版（無料の 6 版は既存の秘密で使い切っている前提） | 約 9 円/版（`put_secret.sh` の注記） | 18 円 |
| Secret Manager のアクセス | 月 24 回（8 回 × 再試行 3）× 4 秘密 ≒ 100 回 | 1 万回まで無料 | 0 円 |
| Cloud Run Jobs | 月 24 回 × 60 秒 × 1 vCPU・0.5GiB ≒ 1,440 vCPU 秒 | 2.9 円/千 vCPU 秒（無料枠は月の前半で尽きる前提） | 約 4 円 |
| GCS | 1 回あたり読み 5・書き 5 程度 × 24 回 ＋ 見張り 30 回 | Class A 0.8 円/千回 | 1 円未満 |
| Artifact Registry | rotator は `live-jobs` と同じ版を使う | — | 0 円 |
| 外への転送 | Job は東京の中。オーナーの取り出しは月 1 回・1KB | — | 0 円 |
| **合計** | | | **約 38 円/月** |

- 依頼どおり Scheduler を 3 本にすると、+30 円になる。1 本にまとめる。
- API（Cloud Run）は Secret Manager を読まない。照合は名簿のハッシュだけなので、起動ごと（最悪 96 回/日）の掛け算は乗らない。名簿の更新は月 3 回増えるだけで、API の取り込みは目録（`auth.recent.json`）による既存の仕組みのまま。
- 台帳の 12 月の見込み（約 1,250〜1,480 円）に足しても、2,000 円の上限に収まる。
- 単価は課金の実績（台帳）によるもので、公式の料金ページとは照合できていない（本文を取得できなかった）。「無効（disabled）の版も課金されるか」も未照合のため、古い版は disable でなく destroy する。
- GCP の外の費用: note の予約投稿には noteプレミアム（月 500 円）が要る。

## 5. 戻し方

- **切り替えの後に戻す**（新しいパスワードに問題があった）: `gcloud run jobs execute keiba-login-rotate --args=-m,src.scripts.ops.shared_login,rollback` を流す。
  - 名簿の `prev_salt`・`prev_hash` を今のハッシュに戻し、`rotation_month` を前の月にする（世代の条件つき）。
  - Secret Manager は、残してある前の版の中身を、新しい版として足す。
  - 会員は先月の note のパスワードで入れる。オーナーは note で知らせる。
- **切り替えが途中で止まった**: 同じ Job をもう一度流す。名簿が確定していれば Secret Manager を追いつかせるだけ。確定していなければ最初からやる。
- **次の版が無いまま 1 日になった**: 切り替わらない（先月のパスワードのまま）。オーナーが、準備 → note の訂正 → 切り替え（`run --step prepare|switch`）を手で流す。
- **新方式そのものをやめる**: Scheduler を止め、crontab を入れ直す。今の名簿のハッシュは残るので、ログインは止まらない。

## 6. 日程と前提

**推奨: 2027-01-01 から**

| 日 | やること | 誰 |
|---|---|---|
| 承認の後 | 実装・テスト・CI・critic のレビュー・develop へ push、stg で確かめる | member-rotation |
| members-purge の push の後 | CI の検査（3.7）を入れる | member-rotation |
| 12 月の前半（週末を避ける） | master への昇格（ユーザーの指示）。jobs のイメージは昇格でしか作られない（`promote.yml`） | ユーザー |
| 昇格の後 | 7 章の GCP の作業（Scheduler は PAUSED で作る）。ペッパーの登録（3.5） | ユーザー（root） |
| 12/02 以降 | 旧経路の停止 → 確かめ → Scheduler の再開（3.6） | ユーザー |
| 12/25 06:00 | 準備（1 月の次の版） | Job |
| 12/28 06:00 | 下書きの通知 → オーナーが note に貼り、1/01 06:00 に予約投稿する | Job → オーナー |
| 1/01 06:00 | 切り替え | Job |

- 12/01 に間に合わせるには、11/24 までに master への昇格と 7 章の作業が要る。11/21・22 は週末の運用と重なり、間に合わない見込みが高い。
- 12 月は今の経路（ホストの cron・派生の値・平文の Slack 通知）のまま。12/01 の前に root の crontab を入れ直して 00:05 JST にするかは、統括役の判断。
  - 入れ直さなければ、12/01 09:05 に切り替わり、そのとき全員のログインが切れる。
  - 入れ直すと 00:05 に切り替わる。
  - どちらも 06:00 ではない。
- 実装の後、Job の動きは stg で確かめる（試験用の名簿の場所・試験用の秘密で）。本番の名簿には、1 月の準備（12/25）まで書かない。

## 7. ユーザーに流してもらう GCP の手順（root の gcloud。実装の後に確定版を docs に書く）

```bash
P=horse-racing-466415; R=asia-northeast1
# 0) 前提の確認（費用の見積もり・取り出しの権限）
gcloud scheduler jobs list --location=$R --project=$P
gcloud secrets list --project=$P
gcloud config get-value account            # ホストの root の認証が人か SA か（3.1）
# 1) 専用の SA と権限
gcloud iam service-accounts create keiba-login-rotator --project=$P
SA=keiba-login-rotator@$P.iam.gserviceaccount.com
#   名簿の書き込み（IAM の条件で名前を絞る。3.7）
gcloud storage buckets add-iam-policy-binding gs://<バケット> --member=serviceAccount:$SA --role=roles/storage.objectUser \
  --condition='title=login-rotate,expression=resource.name.startsWith("projects/_/buckets/<バケット>/objects/ops/mirror/data/local/auth/members.json") || resource.name.startsWith("projects/_/buckets/<バケット>/objects/ops/mirror/_index/auth.") || resource.name.startsWith("projects/_/buckets/<バケット>/objects/ops/login_rotate/")'
#   秘密（パスワードは読み出し・版の追加と destroy。ほかは読み出しだけ）
gcloud secrets create KEIBA_SHARED_LOGIN_PW --replication-policy=user-managed --locations=$R --project=$P
for role in roles/secretmanager.secretAccessor roles/secretmanager.secretVersionManager; do
  gcloud secrets add-iam-policy-binding KEIBA_SHARED_LOGIN_PW --member=serviceAccount:$SA --role=$role --project=$P; done
gcloud secrets add-iam-policy-binding KEIBA_SHARED_LOGIN_PW --member=user:<オーナーのアカウント> --role=roles/secretmanager.secretAccessor --project=$P
bash scripts/gcp/put_secret.sh KEIBA_MEMBERS_PEPPER    # 値の求め方は手順書（3.5）。読めるのは keiba-api と rotator
for s in KEIBA_MEMBERS_PEPPER KEIBA_SHARED_LOGIN_ID SLACK_WEBHOOK_URL; do
  gcloud secrets add-iam-policy-binding $s --member=serviceAccount:$SA --role=roles/secretmanager.secretAccessor --project=$P; done
# 2) Job（イメージは live-jobs と同じ版。配備の手順で live-login-rotate のタグを付ける＝再発防止 4）
bash scripts/gcp/setup_login_rotate_job.sh      # 実装で作る。--set-secrets・max-retries 2・タスク 1・KEIBA_MEMBERS_STORE=gcs
# 3) Scheduler（keiba-g- で始めない。PAUSED で作る）
gcloud scheduler jobs create http keiba-login-rotate --location=$R --project=$P \
  --schedule="0 6 1,25-31 * *" --time-zone=Asia/Tokyo \
  --uri="https://run.googleapis.com/v2/projects/$P/locations/$R/jobs/keiba-login-rotate:run" --http-method=POST \
  --oauth-service-account-email=keiba-scheduler@$P.iam.gserviceaccount.com
gcloud scheduler jobs pause keiba-login-rotate --location=$R --project=$P
# 4) 旧経路を止めて確かめた後に再開（3.6）
sudo bash scripts/cron/install_shared_login_cron.sh --remove
gcloud scheduler jobs resume keiba-login-rotate --location=$R --project=$P
```

- 配備: `scripts/ci/deploy_gcp.sh` の `jobs` の段で、rotator の Job のイメージを `JOBS_IMAGE` に更新し、`live-login-rotate` のタグを付ける。この処理は `ENABLE_JOBS_DEPLOY` の条件の外に置く（条件が偽でも、使用中の版が自動掃除で消えないように）。
- バケットの名前・Scheduler の SA の権限（`run.jobs.run`）は、実装のときに既存の設定（`scripts/gcp/setup_project.sh`）から確かめて埋める。

## 8. note の下調べ（読むだけ。note には触っていない）

- 予約投稿は公式の機能（Web のみ・noteプレミアム）: https://www.help-note.com/hc/ja/articles/360000279862
- 一般向けの公式の API は無い。法人向けの「note API」（2026-10）は下書きを送るだけで、公開は画面で人が行う: https://www.oricon.co.jp/pressrelease/3027859/
- 非公式の API（`note.com/api/v1` など）は勧めない。
  - `https://note.com/robots.txt` が `/api/*` を Disallow している（取得して確認）。
  - 規約には自動化を名指しで禁じる条は見当たらない、という報告がある。ただし「本サービスの障害となる行為」（総則 13.1.7）と「運営に支障」（クリエイター規約 10.1.8）で、通知なく止められうる（二次情報: https://zenn.dev/kiwa_san_san/articles/3ca5cc0996a1b5）。
  - 認証の変更で動かなくなった例がある: https://qiita.com/akiraak/items/0b7de8642cbf72d81271
- **未照合**: 規約の原文（https://terms.help-note.com/hc/ja/articles/44943817565465）とヘルプは 403 で読めなかった。条番号は二次情報による。
- 「有料記事でパスワード（外部サービスの利用権）を売る形」を直接扱う条は、見つからなかった。外部への誘導の仕方によっては削除・停止になった例がある、とする解説はある（https://www.sales-dx.jp/blog/note-terms-of-service-violation）。**採る前に、note の問い合わせ窓口に確かめることを勧める**。
- 将来の完全な自動化の案:

| 案 | 危なさ |
|---|---|
| 公式の API を待つ | 低い。今は法人向けで、公開は人が行う |
| 非公式の API | 高い。robots.txt に反し、予告なく壊れ、通知なく止められうる |
| ブラウザの自動操作 | 中〜高。「運営に支障」に当たりうる。画面が変わると壊れる。note のログイン情報の管理も要る |
| 自前の会員ページ・メール＋決済（Stripe 等） | 規約の危なさは最小で、自動にしやすい。決済・特商法の表記・配信の手間が増える（費用は別に見積もる） |

- 推奨: 当面は手で貼る。完全な自動化に進むなら、配布を自前の会員ページかメールに移し、note は告知に残す（判断に推測を含む）。

## 9. レビューの指摘と対応（別エージェント・2026-11-19）

| # | 重さ | 指摘 | 対応 |
|---|---|---|---|
| 1 | 重 | jobs のイメージは master への昇格でしか作られない。今の `live-jobs` には `run` が無い | 6 章: 開始を 1/01 に。昇格を前提に書いた |
| 2 | 重 | `live-jobs` のタグは `ENABLE_JOBS_DEPLOY` の条件の内側にある | 7 章: rotator のタグ `live-login-rotate` を条件の外に置く |
| 3 | 重 | ホストの旧 cron が新しいパスワードを派生の値に戻す。手順の順番も逆だった | 3.6・7 章: Scheduler は PAUSED で作り、cron を外して確かめた後に再開する |
| 4 | 重 | 下書きの後に次の版が作り直されると、note の値とずれる | 3.2・3.3: `draft_month` を置き、下書きの後は作らずに止める。`draft` は名簿と照合してから表示する |
| 5 | 重 | Job の手元には名簿が無い。`KEIBA_MEMBERS_STORE` が無いと手元に書いて成功を返す | 3.3: GCS の `_current()` を使い、`gcs_store_enabled()` が偽なら止める |
| 6 | 中 | 順番で版を destroy すると、二重に起動したときに相手の版を消しうる | 3.1・3.3: 名簿と合わない版だけを消す。判定は `_transact` の中でやり直す |
| 7 | 中 | `DEV_SECRET_KEY` を渡すと、セッションを偽造できる権限になる | 3.5: `KEIBA_MEMBERS_PEPPER` に分ける |
| 8 | 中 | 見張りが名簿の中身とペッパーを要し、`keiba-jobs` に秘密が要る | 3.7: 秘密の無い `status.json` を読む形に |
| 9 | 中 | バケット全体の `objectUser` は広い | 3.7・7 章: IAM の条件を必須にした |
| 10 | 中 | 下書きの日の定義と表がずれていた | 2 章: 「末日の 3 日前」に統一した |
| 11 | 中 | ホストの root の gcloud が誰か不明 | 3.1・7 章の 0: 確かめる手順を足した |
| 12 | 軽 | 並走の包みは `parallel_guard: false` で外せる | 2 章: 専用にする理由を「秘密を `keiba-jobs` に渡さない」に直した |
| 13 | 軽 | 温まったインスタンスでは、旧いセッションが少し残る | 3.3: 「数分以内」と書いた |
| 14 | 軽 | 試験用アカウントの守りは既にある | 1 章: テストは足さない |
| 15 | 軽 | disabled の版の課金は未照合 | 4 章: destroy を選ぶ。未照合と明記した |
