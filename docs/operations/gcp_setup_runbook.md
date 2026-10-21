# GCP 初期設定の手順書（Cloud Shell でコピペして進める）

目的: VPS（UI）+ GCP（API・ジョブ・データ）構成（`architecture_target.md`）へ移るために、**あなたが GCP Console / Cloud Shell で行う作業**を最小の手数にする。
dev PC と私（Claude）は GCP に接続できない（`gcp_guard`）ので、ここは人の作業。コマンドはすべて **Cloud Shell**（Console 右上の `>_`）で動く形にしてある。

全体の順序と他の作業は `migration_runbook.md`。到達試験の記録は `gcp_smoke_results.md`。

| 章 | やること | 所要 | 追加費用 |
|---|---|---:|---:|
| 0 | Cloud Shell でリポジトリを取る | 3 分 | 0 |
| 1 | プロジェクト・請求先・通貨の確認 | 5 分 | 0 |
| 2 | `setup_project.sh`（API・Artifact Registry・サービスアカウント） | 5 分 | 0 |
| 3 | 権限の確認 | 3 分 | 0 |
| 4 | 毎回使う変数の設定（任意） | 1 分 | 0 |
| 5 | Secret Manager に秘密を登録 | 10 分 | 〜¥36/月 |
| 6 | GCS バケットの監査（読み取りだけ） | 5 分 | 0 |
| 7 | 予算アラート（月 2,000 円） | 5 分 | 0 |
| 8 | 課金エクスポート（BigQuery）と日次コスト通知 | 15 分 | 0 |
| 9 | netkeiba 到達試験（3 回） | 各 5 分 | 0 |
| 10 | つまずいたとき | — | — |
| 11 | 完了チェックリスト | — | — |

---

## 0. Cloud Shell でリポジトリを取る
リポジトリは公開（`meguram/keiba-vpn`）。**必要なフォルダだけ**取る（Cloud Shell の容量を使わない）。

```bash
cd ~
git clone --depth 1 --branch develop --filter=blob:none --sparse https://github.com/meguram/keiba-vpn.git
cd keiba-vpn
git sparse-checkout set deploy/gcp scripts/gcp config docs/operations
ls scripts/gcp deploy/gcp
```
- 確認: `scripts/gcp/setup_project.sh` と `deploy/gcp/netkeiba_smoke/run.sh` が見える。
- 失敗したら: `Remote branch develop not found` → 私がまだ push していない。「push して」と伝える。`Permission denied` → URL の打ち間違い。
- 以降のコマンドは `cd ~/keiba-vpn` から始める。更新を取り直すときは `git pull`。

## 1. プロジェクト・請求先・通貨の確認
```bash
gcloud auth list                                   # ログイン中のアカウント
gcloud projects list --format="table(projectId,name)"
gcloud config set project <プロジェクトID>          # 使うプロジェクトを選ぶ
gcloud config get-value project
gcloud billing projects describe "$(gcloud config get-value project)" --format="value(billingAccountName,billingEnabled)"
gcloud billing accounts list --format="table(name,displayName,open)"
```
- 確認: `billingEnabled` が `True`。
- **通貨は Console で確認**: お支払い → 設定（「お支払いアカウントの設定」）の「通貨」。**JPY か USD か**を控える（7 章の予算アラートの通貨がこれと一致している必要がある）。
- リージョンは **`asia-northeast1`（東京）**で固定（他の章のコマンドの既定値）。
- 失敗したら: `billingEnabled` が False → Console でプロジェクトに請求先アカウントを紐付ける。

## 2. 初期設定を 1 本で（API・Artifact Registry・サービスアカウント・権限）
まず何が実行されるかを確認（何も変更しない）:
```bash
cd ~/keiba-vpn
DRY=1 BUCKET=<既存の GCS バケット名> bash scripts/gcp/setup_project.sh
```
問題なければ本番実行（何度流しても同じ状態になる）:
```bash
BUCKET=<既存の GCS バケット名> bash scripts/gcp/setup_project.sh
```
バケット名がわからないときは先に 6 章の一覧を実行する（`BUCKET` なしでも流せるが、GCS の権限だけ付かない。後で `BUCKET=… bash scripts/gcp/setup_project.sh` を再実行すればよい）。

### 2.1 有効化する API
| API | 用途 |
|---|---|
| `run.googleapis.com` | Cloud Run（API・ジョブ） |
| `cloudscheduler.googleapis.com` | 定期起動（cron の司令塔） |
| `cloudtasks.googleapis.com` | 開催日の予約実行（T-60 / T-15） |
| `secretmanager.googleapis.com` | 秘密の保管 |
| `artifactregistry.googleapis.com` | コンテナ置き場 |
| `cloudbuild.googleapis.com` | コンテナのビルド |
| `billingbudgets.googleapis.com` | 予算アラート |
| `bigquery.googleapis.com` | 課金エクスポートの集計 |
| `iam.googleapis.com` / `logging.googleapis.com` | 上の前提 |

### 2.2 サービスアカウントと最小権限（4 つ）
| サービスアカウント | 使う場所 | 付ける権限 | 付けない権限（理由） |
|---|---|---|---|
| `keiba-api` | Cloud Run の API | GCS バケットの `storage.objectUser`（オブジェクトの読み書きのみ）、各秘密の `secretAccessor`（5 章で個別に付与） | バケット管理・プロジェクト全体の読み書き（API が乗っ取られても被害をバケット内に限る） |
| `keiba-jobs` | Cloud Run Jobs（予測・集計・通知） | GCS の `storage.objectUser`、`bigquery.jobUser`（課金集計）、`cloudtasks.enqueuer`（予約を積む）、秘密の `secretAccessor`（Slack・R2） | Cloud Run の管理（ジョブが自分でデプロイできてはいけない） |
| `keiba-scheduler` | Scheduler / Tasks が Jobs・API を起動するときの身元 | `run.invoker`（起動だけ） | それ以外すべて |
| `keiba-deployer` | ビルドとデプロイ（あなたが Cloud Shell で行う間は使わない。自動化するときに使う） | `run.admin`、`artifactregistry.writer`、`cloudbuild.builds.editor`、上の 3 つへの `iam.serviceAccountUser` | Owner/Editor（広すぎる） |

- `Owner` / `Editor` は誰にも付けない。キー（JSON ファイル）は**作らない**: Cloud Run は SA を直接割り当てれば自動で認証される（`src/config/gcp_credentials.py` は `GCS_PRIVATE_KEY` が無ければ ADC に落ちる）。
- 確認:
  ```bash
  gcloud iam service-accounts list --filter="email~^keiba-" --format="table(email,displayName)"
  gcloud services list --enabled --format="value(config.name)" | grep -E "run|scheduler|tasks|secretmanager|artifactregistry|cloudbuild|billingbudgets|bigquery"
  gcloud artifacts repositories list --location asia-northeast1
  ```
- 失敗したら:
  - `PERMISSION_DENIED`: 自分のアカウントに Owner が無い。Console → IAM で確認。
  - `Billing must be enabled`: 1 章に戻る。
  - `Policy ... constraints/iam.disableServiceAccountCreation`（組織ポリシー）: 組織の管理者に解除を依頼。

## 3. 権限の確認（`run.invoker` が Jobs を起動できるか）
`keiba-scheduler` には `run.invoker` だけを付ける。これが **Jobs の起動（`run.jobs.run`）を含むか**を確認する（未確認の前提）:
```bash
gcloud iam roles describe roles/run.invoker --format="value(includedPermissions)" | tr ';' '\n' | grep -E "run\.(jobs|routes)"
```
- `run.jobs.run` が出ればそのまま。**出ない**場合は、Jobs ごとに限定して付ける（プロジェクト全体の `run.developer` は広すぎる）:
  ```bash
  gcloud run jobs add-iam-policy-binding <ジョブ名> --region asia-northeast1 \
    --member "serviceAccount:keiba-scheduler@$(gcloud config get-value project).iam.gserviceaccount.com" --role roles/run.developer
  ```

## 4. 毎回使う変数の設定（任意）
Cloud Shell が切れると消える。7 章などで使う。
```bash
export PROJECT=$(gcloud config get-value project)
export REGION=asia-northeast1
export BILLING_ACCOUNT=$(gcloud billing accounts list --filter=open=true --format="value(name)" --limit 1 | sed 's|.*/||')
echo "$PROJECT $REGION $BILLING_ACCOUNT"
```

## 5. Secret Manager に秘密を登録する
値は**標準入力から**入れる（画面に出さない・コマンド履歴に残さない・引数にしない）。登録と同時に、読めるサービスアカウントを最小限に絞って付与する（`scripts/gcp/put_secret.sh` の表が正）。

| 秘密の名前 | 中身 | 読める SA | 備考 |
|---|---|---|---|
| `SLACK_WEBHOOK_URL`（または `_PROD` / `_STG`） | Slack の Incoming Webhook URL | api, jobs | 環境別に分けるなら `_PROD`/`_STG` |
| `KEIBA_PROXY_KEY` | VPS の nginx が付ける `X-Keiba-Proxy-Key` の値 | api | VPS の `.env`（権限 600）にも同じ値。長いランダム文字列 |
| `KEIBA_HEARTBEAT_KEY` | VPS → `/ops/heartbeat` の認証 | api | |
| `APP_SECRET_KEY` / `JWT_SECRET` / `DEV_SECRET_KEY` / `DEV_PASSWORD` | アプリのセッション署名・開発者ログイン | api | ホストの `.env` と**同じ値**にする（セッションの互換）。Cloud Run へ移すまでは登録だけ |
| `R2_CREDENTIALS` | Cloudflare R2 のキー（JSON 1 つにまとめる） | jobs | 公開スナップショットを置くときだけ |

- **GCP に置かないもの**: netkeiba のログイン（取得は VPS の実行役のみ）、Cloudflare のトンネルトークン・Access のサービストークン（VPS 側）、`GCS_PRIVATE_KEY`（SA を使うので不要）。
- 登録（値の入力は非表示）:
  ```bash
  cd ~/keiba-vpn
  bash scripts/gcp/put_secret.sh SLACK_WEBHOOK_URL
  # → "値（入力は表示されません）:" で貼り付けて Enter
  ```
- 確認（値は出ない）:
  ```bash
  gcloud secrets list --format="table(name,replication.userManaged.replicas[0].location)"
  gcloud secrets versions list SLACK_WEBHOOK_URL --format="table(name,state,createTime)"
  gcloud secrets get-iam-policy SLACK_WEBHOOK_URL --format="table(bindings.role,bindings.members.flatten())"
  ```
- 差し替え（ローテーション）後は古い版を無効化: `bash scripts/gcp/put_secret.sh SLACK_WEBHOOK_URL --retire`。
- 費用: 有効な版が **6 個を超えた分は 1 版あたり月 約 9 円**（`config/cost_assumptions.json` の案 A は 4 版超過ぶんを見込み済み、最大 約 36 円）。
- 失敗したら: `未登録の名前です` → 名前の打ち間違い。新しい秘密を足すときは `put_secret.sh` の表とこの表を同時に直す。`PERMISSION_DENIED` → `secretmanager.googleapis.com` が有効か（2 章）。

## 6. GCS バケットの監査（読み取りだけ・何も変更しない）
```bash
cd ~/keiba-vpn
bash scripts/gcp/audit_gcs.sh                       # プロジェクト内の全バケットの一覧
bash scripts/gcp/audit_gcs.sh <バケット名>           # リージョン・クラス・版管理・ライフサイクル・権限
DU=1 bash scripts/gcp/audit_gcs.sh <バケット名>      # 使用量も数える（オブジェクトが多いと数分。読み取りのみ）
```
素のコマンドで見たいとき:
```bash
gcloud storage buckets describe gs://<バケット名> --format="yaml(location,location_type,default_storage_class,versioning_enabled,lifecycle_config,uniform_bucket_level_access,public_access_prevention)"
gcloud storage du --summarize --readable-sizes gs://<バケット名>
```
判断の目安と、結果をこの表に貼って私に返す:
| 項目 | 望ましい | 違ったとき |
|---|---|---|
| リージョン | `ASIA-NORTHEAST1`（東京） | 別リージョンだと Cloud Run との間の転送が有料・遅い。移すか、Cloud Run をバケットに合わせる |
| クラス | `STANDARD` | Nearline 等は読むたびに取り出し料金 |
| バージョニング | 要否を決める | 有効だと古い版も保存料金（月 約 345 円/100GB）。生データは不要なら無効 |
| ライフサイクル | 古い生データを Nearline/削除するルール | 無ければ `architecture_target.md` 9 章 7 の決定後に設定 |
| 公開アクセス防止 | `enforced` | `inherited` なら有効化 |
- 結果（バケット名・リージョン・容量・版管理・ライフサイクル）を `gcp_smoke_results.md` の末尾「GCS 監査の記録」に控える。

## 7. 予算アラート（月 2,000 円）
**予算アラートは通知だけで、100% に達しても課金は止まらない**（止めるにはプロジェクトの課金を外す＝全サービス停止になるので、自動停止は付けない）。暴走防止は Cloud Run の最大インスタンス数・ジョブのリトライ上限・Scheduler の頻度で行う（`architecture_target.md` 6.4）。

通貨で金額が変わる（請求先アカウントの通貨に**必ず合わせる**）:
```bash
# 4 章の変数（PROJECT, BILLING_ACCOUNT）を使う。請求先が複数あるときは Console で正しい ID を確認して BILLING_ACCOUNT を直す
# 請求先が JPY の場合
gcloud billing budgets create --billing-account="$BILLING_ACCOUNT" --display-name="keiba 月2000円" \
  --budget-amount=2000JPY --filter-projects="projects/$PROJECT" \
  --threshold-rule=percent=0.5 --threshold-rule=percent=0.8 --threshold-rule=percent=1.0 \
  --threshold-rule=percent=1.0,basis=forecasted-spend
# 請求先が USD の場合（150 円/ドルで 約 13 ドル）は --budget-amount=13USD に変える
```
- 確認: `gcloud billing budgets list --billing-account="$BILLING_ACCOUNT"`。通知は請求先アカウントの管理者へメールが既定。Slack にも出したいときは、日次コストレポート（8 章）の月末見込みと `[GCPコスト警告]` が代わりをする。
- 失敗したら: `currency code mismatch` → 通貨を逆にして再実行。`billingbudgets API has not been used` → 2 章をやり直す。`Permission denied` → 請求先アカウントの管理者権限が必要（プロジェクトの Owner とは別）。

## 8. 課金エクスポート（BigQuery）と日次コスト通知
実績の取得には BigQuery への課金エクスポートが必要（Console 操作のみ。コードでは自動化できない）。**有効にした時点からのデータしか溜まらない**（過去分は入らない）ので、早く有効にするほど良い。

1. データセットを作る（東京）:
   ```bash
   bq --location=asia-northeast1 mk --dataset "$(gcloud config get-value project):billing_export"
   ```
2. Console → お支払い → **課金データのエクスポート** → 「BigQuery エクスポート」→ **標準の使用料金**の「設定を編集」→ プロジェクトと `billing_export` を選んで保存。
3. 数時間〜1 日後にテーブルができる。名前を確認して控える:
   ```bash
   bq ls "$(gcloud config get-value project):billing_export"
   # gcp_billing_export_v1_XXXXXX_XXXXXX_XXXXXX が出る
   ```
4. 日次ジョブ（`keiba-jobs`）が読めるようにする（`bigquery.jobUser` は 2 章で付与済み。**データセットの閲覧**を足す）:
   Console → BigQuery → `billing_export` → 共有 → 権限 → `keiba-jobs@…` に **BigQuery データ閲覧者**。
5. ジョブの環境変数に `GCP_BILLING_BQ_TABLE=<プロジェクト>.billing_export.gcp_billing_export_v1_…` を設定し、コマンドに **`--watch`** を付ける:
   `python -m src.scripts.cloud_jobs.gcp_daily_cost_report --watch`（登録の雛形: `gcp-cloud-run-jobs.md` の 15）。毎日 07:00 JST。

通知の内容（`--watch` 付き）:
```
[GCP日次コストレポート] 2026-11-03: 合計 $0.42（Cloud Storage: $0.30, …）
今月累計 ¥130 / 月末見込み ¥1,250（予算 ¥2,000 の 7%）
フェーズ3「予測生成・集計・通知の GCP 化」の想定 ¥0〜¥1,400/月 → 範囲内
```
- 「想定」は `config/migration_phases.json` の `current_phase`（**フェーズを進めたら手で書き換える**。`migration_runbook.md`）と、案 A のサービス別上限（`python -m src.scripts.ops.cost_estimate`）。状態は「範囲内 / 上限超過 / 下限未満（取得・ジョブが動いていない可能性）」。
- 別メッセージ `[GCPコスト警告]` が出る条件（しきい値は `config/migration_phases.json` の `spike`）:
  - サービス別: 当日が**前日または直近 7 日平均の 3 倍以上**かつ増分 **30 円以上**（前日が 5 円未満なら 5 円で割る＝小さな数字のノイズを避ける）
  - 日次の合計が **200 円以上**（予算 ÷ 30 日の 3 倍）
  - 月末見込みが案 A のサービス別上限を 30 円以上超える（月の 3 日目以降）
- 金額は**クレジット（無料枠）差し引き後の請求額**。USD の請求先は 1 ドル = `config/cost_assumptions.json` の為替で円に換算する。先頭の `[GCP日次コストレポート]` 行の記号は従来のまま `$` 固定（円の請求先では数字が円）。
- 失敗したら: 通知に「取得失敗（GCP_BILLING_BQ_TABLE が未設定…）」→ 5 の環境変数。「Not found: Table」→ 3 のテーブル名の誤り・まだ作られていない。「Access Denied」→ 4 の閲覧権限。「想定との比較は取得失敗」→ 基本の日次通知は届いているので、履歴クエリの権限（4）を確認。

## 9. netkeiba 到達試験（GCP の IP から届くか）
`architecture_target.md` 7.4 の試験。**公開ページのみ・ログインなし・30 リクエスト（約 3 分）を 3 回**（日と時間帯を変える）。VPS（ConoHa）は 30/30 成功済み。

1. 実行前に確認: **ホストの取得ジョブが止まっていること**（DEC-023: netkeiba へ同時にアクセスしない）。
   ```bash
   ssh keiba-remote "pgrep -af '[a]uto_scrape' || echo 動いていない"
   ```
2. 推奨の実行枠（JST）:
   | 枠（引数） | いつ | 狙い |
   |---|---|---|
   | `weekday_noon` | 平日の昼（12:00〜14:00） | 通常時の混雑 |
   | `raceday_morning` | 開催日（土日）の朝（7:00〜9:00。まだ予測・オッズ取得が走る前） | 開催日の負荷 |
   | `night` | 夜間（23:00〜翌 2:00。メンテナンス窓 3:00〜4:00 は避ける） | 夜間の取得想定 |
3. Cloud Shell で（1 回ずつ。間を空けて 3 回）:
   ```bash
   cd ~/keiba-vpn/deploy/gcp/netkeiba_smoke
   bash run.sh weekday_noon
   ```
   ビルド → ジョブ作成 → 実行 → 要約までを 1 本で行い、最後に `RESULT: PASS` か `RESULT: FAIL …` と、送信元 IP、`gcp_smoke_results.md` に貼る 1 行を出す。
4. 3 回終わったら `CLEANUP=1 bash run.sh night` のようにして最後にジョブを削除（または `gcloud run jobs delete netkeiba-smoke --region asia-northeast1`）。
5. **合格条件**: 3 回とも成功率 93%（28/30）以上、403/429 が 0 件、遮断を示す語が 0 件。**403/429 が 1 回でも出たら試験は自動で中止され FAIL**（これ以上は叩かない）。
6. 結果の行を `gcp_smoke_results.md` に貼って私に知らせる。不合格なら取得は VPS の実行役に集約する（`architecture_target.md` 5.1）。
- 費用: Cloud Build 無料枠（2,500 分/月）・Artifact Registry 0.5GB・Cloud Run Jobs 数十 vCPU 秒 → 0 円。
- 失敗したら:
  - `builds submit` が `PERMISSION_DENIED` / `storage.objects.get`: Cloud Build の既定 SA に権限が無い。`PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')` を取り、`gcloud projects add-iam-policy-binding $(gcloud config get-value project) --member serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com --role roles/cloudbuild.builds.builder` を実行して再試行。
  - `RESULT: UNKNOWN`: ログの反映待ち。数十秒後に `gcloud run jobs executions list --job netkeiba-smoke --region asia-northeast1` で状態を見る。コンテナの終了コード（0=合格）が正。
  - `egress_ip` が空: IP 取得サービスに届かなかっただけ。試験自体の合否には影響しない。

## 10. つまずいたとき（共通）
| 症状 | 原因と対処 |
|---|---|
| `gcloud: command not found` | Cloud Shell 以外で実行している。Console 右上の `>_` から開く |
| 「Cloud Shell が切れた」 | 変数（`PROJECT` など）は消える。`gcloud config set project` からやり直す。スクリプトは冪等なので再実行してよい |
| API の有効化が長い | 初回は 1〜2 分かかることがある。そのまま待つ |
| 組織ポリシーで拒否 | 個人の無料枠のプロジェクトなら通常は出ない。出たら組織管理者の設定 |
| 想定外の課金が出た | 8 章の `[GCPコスト警告]` と Console → お支払い → レポートで、サービス別・SKU 別に見る。Cloud Run は最大インスタンス数とリトライ、Scheduler は頻度を疑う |

## 11. 完了チェックリスト
- [ ] 1: プロジェクト・請求先・通貨（JPY/USD）を控えた
- [ ] 2: `setup_project.sh` が最後まで通り、確認の表が出た
- [ ] 3: `run.invoker` に `run.jobs.run` が含まれる（または Jobs 単位で付与）
- [ ] 5: 秘密を登録（少なくとも `SLACK_WEBHOOK_URL`、`KEIBA_PROXY_KEY`）
- [ ] 6: GCS の監査結果を控えた
- [ ] 7: 予算アラートを作った（通貨が一致）
- [ ] 8: 課金エクスポートを有効にし、翌日に `bq ls` でテーブルを確認した
- [ ] 9: 到達試験を 3 回行い、結果を `gcp_smoke_results.md` に貼った

次: `migration_runbook.md` のフェーズ 2（API を Cloud Run へ）。
