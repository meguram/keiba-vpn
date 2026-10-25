# D2 実行手順（Cloud Shell。master 昇格の後に使う最終版）

> `gcp_jobs_migration.md` 4 章の実行用。1 コマンドずつ実行し、**確認コマンドの出力が期待どおりなら次へ**。失敗したら、その行の「戻す」を実行して止める。
> 並走（`--parallel`）の間、Jobs の出力は GCS の `state/` にだけ上がり、画面が読む `ops/mirror` には**上がらない**（ホストの出力を上書きしない）。ここまでの手順で本番の表示は変わらない。

## 0. 準備（読み取りだけ）
```bash
export PROJECT_ID=$(gcloud config get-value project) GCS_BUCKET=magu-keiba-horse-racing-ai REGION=asia-northeast1
# 昇格で CI が push した Jobs のイメージ（<8桁> = 昇格したコミットの先頭 8 桁。GitHub Actions の promote の要約に出る）
export IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/keiba-vpn/jobs:<8桁>
cd ~/keiba-vpn && git fetch origin && git checkout <昇格した master のコミット>     # スクリプトとジョブ定義をイメージと同じ版にそろえる
```
確認（すべて出力があること）:
```bash
gcloud artifacts docker images describe $IMAGE --format='value(image_summary.digest)'                      # イメージが実在する
gcloud iam service-accounts list --format='value(email)' | grep -E 'keiba-(jobs|scheduler)@'              # 2 本ある
gcloud secrets describe SLACK_WEBHOOK_URL --format='value(name)'                                           # Jobs が渡す秘密がある
gcloud storage ls gs://$GCS_BUCKET/models/                                                                 # bundle_v6e/ stage_v6e/ tl_reference_v5/ serving_bundle/ lap_models/ final_odds/ fitted_artifacts/
```
中身の確認だけ（実行しない）: `bash scripts/gcp/deploy_cloud_run_jobs.sh show all --parallel`

## ④ Cloud Tasks のキュー
```bash
bash scripts/gcp/deploy_cloud_run_jobs.sh apply queue --yes
```
- 確認: `gcloud tasks queues describe keiba-predict --location $REGION --format='value(state,rateLimits.maxConcurrentDispatches)'` → `RUNNING  3`
- 戻す: `gcloud tasks queues pause keiba-predict --location $REGION`（削除はしない。同名は 7 日再作成できない）

## ⑤ Run Job（s / m / l の 3 つ。同じイメージ）
```bash
bash scripts/gcp/deploy_cloud_run_jobs.sh apply jobs --parallel --yes      # 上で export した IMAGE が使われる
```
- 確認: `gcloud run jobs list --region $REGION --format='table(name,spec.template.spec.template.spec.containers[0].image.basename(),spec.template.spec.template.spec.containers[0].resources.limits.memory)'`
  → `keiba-dispatch-s 512Mi / -m 2Gi / -l 8Gi`、イメージが上の 8 桁。環境変数に `KEIBA_PARALLEL_RUN=1` があること:
  `gcloud run jobs describe keiba-dispatch-l --region $REGION --format=json | grep -c KEIBA_PARALLEL_RUN`（1 以上）
- 戻す: `for s in s m l; do gcloud run jobs delete keiba-dispatch-$s --region $REGION --quiet; done`

## ⑥ IAM（Run Job を作った後に実行。2 本）
```bash
PROJECT=$PROJECT_ID
for j in keiba-dispatch-s keiba-dispatch-m keiba-dispatch-l; do
  gcloud run jobs add-iam-policy-binding $j --region $REGION --member "serviceAccount:keiba-scheduler@$PROJECT.iam.gserviceaccount.com" --role roles/run.developer
done
gcloud iam service-accounts add-iam-policy-binding keiba-scheduler@$PROJECT.iam.gserviceaccount.com \
  --member "serviceAccount:keiba-jobs@$PROJECT.iam.gserviceaccount.com" --role roles/iam.serviceAccountUser
```
- 確認: `gcloud run jobs get-iam-policy keiba-dispatch-l --region $REGION | grep keiba-scheduler`
- 戻す: 同じコマンドの `add-iam-policy-binding` を `remove-iam-policy-binding` に替える。

## ⑦ event-pump の動作確認（ready のジョブ。副作用は小さい）
```bash
bash scripts/gcp/deploy_cloud_run_jobs.sh run-now job event-pump --yes
```
- 確認（実行記録）: `gcloud storage cat gs://$GCS_BUCKET/ops/job_runs/event-pump/latest.json | python3 -m json.tool | grep -E '"status"|"reason"|duration_sec'` → `"status": "ok"`
- 失敗したら: ログ `gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="keiba-dispatch-m"' --limit 30 --format='value(textPayload)' --freshness=30m` を 37 に渡す。直すまで次へ進まない。

## ⑨ predict-race を 1 レースだけ手動で（並走なので本番の表示は変わらない）
手順の正は `gcp_user_steps_20261026.md` の B5。要点: `gcloud run jobs execute --args` は `--arg` を繰り返せない（`--args` の短縮と解釈される）ので、Cloud Run Admin API の REST（Cloud Tasks と同じ形）で起動する。`KEY=VALUE` を位置引数で受ける版（次の昇格以降）なら `ALLOW_UNREADY=1 bash scripts/gcp/deploy_cloud_run_jobs.sh run-now job predict-race race_id=… date=… --yes`（オプションは job_id より前）。
確認は `ops/job_runs/predict-race/latest.json`（`status`・`parallel`・`memory_peak_mib`）と `state/data/local/inference/predictions_v6/<日>/<race_id>.parquet`。戻すのは `state/` の該当日を削除するだけ。

## 通ったら（37 経由で判断）
1. `python -m src.ops.job_map ready predict-race --note "<確認した内容>"`（コミットして次の昇格に入れる）。
2. 同じ日の予測をホストと突き合わせる: `python -m src.scripts.ops.parity_check auto --date 20261025`。
3. Scheduler は、ready のジョブがあるグループだけ作られる: `bash scripts/gcp/deploy_cloud_run_jobs.sh apply scheduler --parallel --yes`（14 本以内）。
4. 切替（`--parallel` を外して `apply jobs` を再実行）は、数日の並走で `cron_cutover_status` が合格してから。そのとき初めて `mirror_out` が有効になり、Jobs の予測が画面に出る。

## 費用の目安
`predict-race` は 1 回約 1 分・8GiB×2vCPU、開催日 36〜72 回。Cloud Run Jobs の無料枠（240,000 vCPU 秒／450,000 GiB 秒 毎月）を含めて月 100〜300 円程度の見込み（実測ではない）。⑨の `memory_peak_mib` が小さければ 4GiB に下げて半分にできる。
