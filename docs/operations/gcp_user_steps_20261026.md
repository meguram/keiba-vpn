# 2026-10-26 Cloud Shell で実行する手順（最終版。1 コマンドずつ。確認コマンドと戻し方つき）

> 上から順に。**各コマンドの確認が期待どおりなら次へ**。失敗したら「戻す」を実行して止め、出力を keiba-vpn-37 に渡す。
> 条件つき IAM は、式が長いので `scripts/gcp/make_iam_condition.sh` で YAML を作ってから `--condition-from-file` で渡す（読み書きと一覧で条件が別々に要り、手で書くと片方を落としやすい）。
> 並走（`--parallel`）の間、Jobs の出力は `state/` にだけ上がり、本番の表示は変わらない。ここまでの手順で、画面・API の挙動は変わらない。

## 0. 準備（Cloud Shell。最初の 1 回）
```bash
export PROJECT_ID=$(gcloud config get-value project) GCS_BUCKET=magu-keiba-horse-racing-ai REGION=asia-northeast1
export PROJECT=$PROJECT_ID BUCKET=$GCS_BUCKET
```
```bash
[ -d ~/keiba-vpn/.git ] || git clone https://github.com/meguram/keiba-vpn.git ~/keiba-vpn
```
```bash
cd ~/keiba-vpn && git fetch origin && git checkout master && git pull --ff-only origin master && git log --oneline -1
```
（master 昇格の前なら、上の `master` を `stg` に替えてよい。スクリプトとジョブ定義だけを使う。）

## A. イメージが無くても進められるもの

### A1. Cloud Tasks のキュー
```bash
bash scripts/gcp/deploy_cloud_run_jobs.sh apply queue --yes
```
- 確認: `gcloud tasks queues describe keiba-predict --location $REGION --format='value(state,rateLimits.maxConcurrentDispatches)'` → `RUNNING  3`
- 戻す: `gcloud tasks queues pause keiba-predict --location $REGION`（削除はしない。同名は 7 日再作成できない）

### A2. keiba-jobs が keiba-scheduler の身元でタスクを積める（act-as）
```bash
gcloud iam service-accounts add-iam-policy-binding keiba-scheduler@$PROJECT.iam.gserviceaccount.com --member "serviceAccount:keiba-jobs@$PROJECT.iam.gserviceaccount.com" --role roles/iam.serviceAccountUser
```
- 確認: `gcloud iam service-accounts get-iam-policy keiba-scheduler@$PROJECT.iam.gserviceaccount.com | grep -B1 -A2 keiba-jobs`
- 戻す: 同じコマンドの `add-iam-policy-binding` を `remove-iam-policy-binding` に替える。

### A3. keiba-publisher（models/ だけに書ける SA。ホストの運用の書き込み鍵を外す前提）
```bash
gcloud iam service-accounts create keiba-publisher --display-name "keiba publisher (models/ only)"
```
```bash
BUCKET=$BUCKET bash scripts/gcp/make_iam_condition.sh models/ models-write > /tmp/cond_publisher.yaml && cat /tmp/cond_publisher.yaml
```
```bash
gcloud storage buckets add-iam-policy-binding gs://$BUCKET --member "serviceAccount:keiba-publisher@$PROJECT.iam.gserviceaccount.com" --role roles/storage.objectUser --condition-from-file /tmp/cond_publisher.yaml
```
- 確認: `gcloud storage buckets get-iam-policy gs://$BUCKET --format=json | grep -B3 -A6 keiba-publisher`（条件つきで 1 件）
- 鍵（ホストに置く。**画面に出さない・チャットに貼らない**）: `gcloud iam service-accounts keys create /tmp/publisher.json --iam-account keiba-publisher@$PROJECT.iam.gserviceaccount.com` → ホストの `~/.config/keiba-gcp/publisher.json`（`chmod 600`）へ安全に移し、Cloud Shell 側は `shred -u /tmp/publisher.json` で消す。続きは `publisher_sa.md` の 4〜5（`ops/` に書くと 403 になる確認）。
- 戻す: `gcloud iam service-accounts delete keiba-publisher@$PROJECT.iam.gserviceaccount.com`（鍵も消える）

### A4. keiba-monitor（監視ポータル用の読み取り専用 SA。デプロイは B6）
```bash
gcloud iam service-accounts create keiba-monitor --display-name "keiba monitor (read-only ops/)"
```
```bash
BUCKET=$BUCKET bash scripts/gcp/make_iam_condition.sh ops/ ops-read > /tmp/cond_monitor.yaml && cat /tmp/cond_monitor.yaml
```
```bash
gcloud storage buckets add-iam-policy-binding gs://$BUCKET --member "serviceAccount:keiba-monitor@$PROJECT.iam.gserviceaccount.com" --role roles/storage.objectViewer --condition-from-file /tmp/cond_monitor.yaml
```
```bash
gcloud secrets add-iam-policy-binding KEIBA_PROXY_KEY --member "serviceAccount:keiba-monitor@$PROJECT.iam.gserviceaccount.com" --role roles/secretmanager.secretAccessor
```
- 確認: `gcloud storage buckets get-iam-policy gs://$BUCKET --format=json | grep -B3 -A6 keiba-monitor` / `gcloud secrets get-iam-policy KEIBA_PROXY_KEY | grep keiba-monitor`
- 戻す: `gcloud iam service-accounts delete keiba-monitor@$PROJECT.iam.gserviceaccount.com`

### A5. keiba-fetch（VPS の取得役。`vps_agent.md` 5.5。鍵は VPS の `/etc/keiba-fetch/sa.json` に置く）
SA を作る:
```bash
gcloud iam service-accounts create keiba-fetch --display-name "keiba fetch agent (VPS)"
```
タスク表（`ops/fetch_tasks/`）と、イベント（`ops/events/`）に書けるようにする（読み書き・一覧。接頭辞の外は不可）:
```bash
BUCKET=$BUCKET bash scripts/gcp/make_iam_condition.sh ops/fetch_tasks/ fetch-tasks > /tmp/cond_ft.yaml && gcloud storage buckets add-iam-policy-binding gs://$BUCKET --member "serviceAccount:keiba-fetch@$PROJECT.iam.gserviceaccount.com" --role roles/storage.objectUser --condition-from-file /tmp/cond_ft.yaml
```
```bash
BUCKET=$BUCKET bash scripts/gcp/make_iam_condition.sh ops/events/ fetch-events > /tmp/cond_ev.yaml && gcloud storage buckets add-iam-policy-binding gs://$BUCKET --member "serviceAccount:keiba-fetch@$PROJECT.iam.gserviceaccount.com" --role roles/storage.objectUser --condition-from-file /tmp/cond_ev.yaml
```
影モード（ホストが取得を続けたまま VPS の取得を見届ける間）は、`shadow/` と `ops/fetch_tasks_shadow/` だけに書ける:
```bash
BUCKET=$BUCKET bash scripts/gcp/make_iam_condition.sh shadow/ fetch-shadow > /tmp/cond_sh.yaml && gcloud storage buckets add-iam-policy-binding gs://$BUCKET --member "serviceAccount:keiba-fetch@$PROJECT.iam.gserviceaccount.com" --role roles/storage.objectUser --condition-from-file /tmp/cond_sh.yaml
```
```bash
BUCKET=$BUCKET bash scripts/gcp/make_iam_condition.sh ops/fetch_tasks_shadow/ fetch-tasks-shadow > /tmp/cond_fts.yaml && gcloud storage buckets add-iam-policy-binding gs://$BUCKET --member "serviceAccount:keiba-fetch@$PROJECT.iam.gserviceaccount.com" --role roles/storage.objectUser --condition-from-file /tmp/cond_fts.yaml
```
本番のデータは**読み取りだけ**（影モードの比較用。書き込み権限は付けない）:
```bash
gcloud storage buckets add-iam-policy-binding gs://$BUCKET --member "serviceAccount:keiba-fetch@$PROJECT.iam.gserviceaccount.com" --role roles/storage.objectViewer
```
鍵（VPS に置く。**画面に出さない**）:
```bash
gcloud iam service-accounts keys create /tmp/fetch-sa.json --iam-account keiba-fetch@$PROJECT.iam.gserviceaccount.com
```
→ 安全に VPS の `/etc/keiba-fetch/sa.json`（`chmod 640 root:keibafetch`）へ移し、`shred -u /tmp/fetch-sa.json`。
- 確認: `gcloud storage buckets get-iam-policy gs://$BUCKET --format=json | grep -c keiba-fetch`（条件つき 4 件＋閲覧 1 件。同じ SA の行数で数える）
- 戻す: `gcloud iam service-accounts delete keiba-fetch@$PROJECT.iam.gserviceaccount.com`
- 注意: 影モードをやめて取得の担当を移すときは、**本番のデータの接頭辞への書き込み**（`race_shutuba` 等のカテゴリ）が別に要る。そのときに範囲を決めて、この手順に足す（今は付けない）。

## B. master 昇格（承認）の後。Jobs のイメージができてから
昇格の要約（GitHub Actions の promote）に出る、昇格コミットの先頭 8 桁を `<8桁>` に入れる。

### B1. イメージの実在確認（最初に必ず）
```bash
export IMAGE=$REGION-docker.pkg.dev/$PROJECT/keiba-vpn/jobs:<8桁> && gcloud artifacts docker images describe $IMAGE --format='value(image_summary.digest)'
```
- 期待: `sha256:…` が出る。**`Image not found` なら止めて keiba-vpn-37 へ**（CI の jobs ビルドが失敗している）。

### B2. Run Job（s / m / l。同じイメージ。並走モード）
```bash
git checkout <昇格したコミット> && bash scripts/gcp/deploy_cloud_run_jobs.sh apply jobs --parallel --yes
```
- 確認: `gcloud run jobs list --region $REGION --format='table(name,spec.template.spec.template.spec.containers[0].resources.limits.memory)'` → `s 512Mi / m 2Gi / l 8Gi`。`gcloud run jobs describe keiba-dispatch-l --region $REGION --format=json | grep -c KEIBA_PARALLEL_RUN` が 1 以上。
- 戻す: `for s in s m l; do gcloud run jobs delete keiba-dispatch-$s --region $REGION --quiet; done`

### B3. Run Job の起動権限（Run Job を作った後）
```bash
for j in keiba-dispatch-s keiba-dispatch-m keiba-dispatch-l; do gcloud run jobs add-iam-policy-binding $j --region $REGION --member "serviceAccount:keiba-scheduler@$PROJECT.iam.gserviceaccount.com" --role roles/run.developer; done
```
- 確認: `gcloud run jobs get-iam-policy keiba-dispatch-l --region $REGION | grep keiba-scheduler`
- 戻す: `add-iam-policy-binding` を `remove-iam-policy-binding` に替える。

### B4. event-pump の動作確認
```bash
bash scripts/gcp/deploy_cloud_run_jobs.sh run-now job event-pump --yes
```
- 確認: `gcloud storage cat gs://$BUCKET/ops/job_runs/event-pump/latest.json | python3 -m json.tool | grep -E '"status"|"reason"'` → `"status": "ok"`

### B5. predict-race を 1 レースだけ（並走なので本番の表示は変わらない）
定義の確認だけ（何も書かない）:
```bash
gcloud run jobs execute keiba-dispatch-l --region $REGION --wait --args=-m,src.ops.job_dispatch,job,predict-race,--allow-unready,--dry-run,--arg,race_id=202608040901,--arg,date=20261025
```
実行:
```bash
gcloud run jobs execute keiba-dispatch-l --region $REGION --wait --args=-m,src.ops.job_dispatch,job,predict-race,--allow-unready,--arg,race_id=202608040901,--arg,date=20261025
```
- 確認 1: `gcloud storage cat gs://$BUCKET/ops/job_runs/predict-race/latest.json | python3 -m json.tool | grep -E '"status"|"reason"|duration_sec|memory_peak_mib|memory_limit_mib|"parallel"'` → `status ok`、`parallel true`。**`memory_peak_mib` を keiba-vpn-37 に伝える。**
- 確認 2: `gcloud storage ls gs://$BUCKET/state/data/local/inference/predictions_v6/20261025/` に `202608040901.parquet`
- 失敗の典型: `FileNotFoundError … page_reference/tables/…`（足りない表。ファイル名を伝える）、`Out of memory`（メモリ。伝える）。
- 戻す: `gcloud storage rm -r gs://$BUCKET/state/data/local/inference/predictions_v6/20261025/`

### B6. 監視ポータル（keiba-monitor）のデプロイ（A4 のあと。API と同じイメージ。`keiba-api:<8桁>`）
```bash
export IMAGE=$REGION-docker.pkg.dev/$PROJECT/keiba-vpn/keiba-api:<8桁> && bash scripts/gcp/deploy_cloud_run_monitor.sh
```
（上は表示だけ。内容を確認して）:
```bash
bash scripts/gcp/deploy_cloud_run_monitor.sh --apply
```
- 確認: `curl -s $(gcloud run services describe keiba-monitor --region $REGION --format='value(status.url)')/healthz` → `{"status":"ok"}`
- 戻す: `gcloud run services delete keiba-monitor --region $REGION --quiet`
