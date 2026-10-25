# 監視ポータルの Cloud Run 化（D4）

> 方針（`decoupling_plan.md` 4.6）: 運用の監視は **GCP（Cloud Run・最小 0）に置き、VPS の死活を VPS の外から見る**。ホストの `src/monitor/app.py`（Flask :23900）は研究者向けとして残してよいが、運用の監視には使わない。

## 1. 何を作ったか
- `src/monitor/cloud_app.py`（読み取り専用の Flask。ホストのローカルファイルを一切読まず、GCS の `ops/` の JSON だけを見る）。
  - VPS ハートビート（`ops/heartbeat/latest.json`。`heartbeat_schema.evaluate` で異常を出す）
  - Jobs の最新の実行（`ops/job_runs/<job>/latest.json`。状態・所要・メモリのピーク/上限・並走か）
  - 今日の異常通知（`ops/alerts/<日>/`）
  - 画面・取得役の配布（`ops/releases/<kind>/latest.json` と VPS が報告する `applied/vps.json`。版ずれが一目で分かる）
  - 成果物の一致（`ops/parity/<日>/`）と切替判定（`ops/cutover/status.json`）
  - `/` が表（全体の信号つき）、`/api/overview` が同じ内容の JSON、`/healthz` が起動確認（鍵なし）。
- 認証は API と同じ共有ヘッダ `X-Keiba-Proxy-Key`（`KEIBA_PROXY_KEY_REQUIRED=1`）。前段の Cloudflare Access / VPS の nginx が付ける。
- 配備は `scripts/gcp/deploy_cloud_run_monitor.sh`（既定は表示のみ、`--apply` で実行）。**イメージは API と同じ**（起動コマンドだけ `uvicorn --factory src.monitor.cloud_app:asgi_app`）。メモリ 512Mi・最大 1 台・最小 0。

## 2. 手順（ユーザー。Cloud Shell。1 つずつ）
1. 読み取り専用の SA を作る（`ops/` だけ読める）:
   ```bash
   PROJECT=$(gcloud config get-value project); BUCKET=<運用のバケット>
   gcloud iam service-accounts create keiba-monitor --display-name "keiba monitor (read-only ops/)"
   # 読み取り(get)は resource.name、一覧(list)は objectListPrefix で別々に条件が要る。両方を式に入れる（条件はファイルで渡す）
   BUCKET=$BUCKET bash scripts/gcp/make_iam_condition.sh ops/ ops-read > /tmp/cond_monitor.yaml
   gcloud storage buckets add-iam-policy-binding gs://$BUCKET \
     --member "serviceAccount:keiba-monitor@$PROJECT.iam.gserviceaccount.com" --role roles/storage.objectViewer --condition-from-file /tmp/cond_monitor.yaml
   ```
2. 秘密 `KEIBA_PROXY_KEY` に、この SA の読み取りを許す（API と同じ秘密を使う）:
   ```bash
   gcloud secrets add-iam-policy-binding KEIBA_PROXY_KEY --member "serviceAccount:keiba-monitor@$PROJECT.iam.gserviceaccount.com" --role roles/secretmanager.secretAccessor
   ```
3. 表示だけで中身を確認 → 配備:
   ```bash
   export PROJECT_ID=$PROJECT GCS_BUCKET=$BUCKET IMAGE=asia-northeast1-docker.pkg.dev/$PROJECT/keiba-vpn/keiba-api:<CI が push した8桁>
   bash scripts/gcp/deploy_cloud_run_monitor.sh            # 表示だけ
   bash scripts/gcp/deploy_cloud_run_monitor.sh --apply
   ```
4. 起動確認: `curl -s $(gcloud run services describe keiba-monitor --region asia-northeast1 --format='value(status.url)')/healthz`（`{"status":"ok"}`）。鍵を付けて `/api/overview` が返ること。
5. 公開の道（VPS と Cloudflare。ここは VPS 側の担当）: `monitor.megu-keiba.com` の向き先を、ホストの :23900 から「VPS の nginx → keiba-monitor の Cloud Run URL（`X-Keiba-Proxy-Key` を付与）」に切り替える。Cloudflare Access の保護は今のまま。切り替えが済んでから、ホスト側の Tunnel のルートを外す。

## 3. 費用
無料枠の範囲の見込み（`decoupling_plan.md` 7 章。リクエストは人が見るときだけ、最小 0、最大 1 台）。読み取りは GCS の小さな JSON が 1 表示あたり数十件（Class B）。

## 4. 未対応（ホストの監視ポータルにあって、ここには無いもの）
ホストの systemd/プロセス・git の状態・ログの tail・スクレイピングのカバレッジ表は、ホスト（や VPS）のローカルを見るため移していない。必要なものは、収集側が GCS に JSON を出す形（ハートビートと同じ）にしてから足す。優先度の高い順: ①取得役（VPS）のタスクの完了/失敗（`ops/fetch_tasks/`）②予測成績（`ops/prediction_results/`）③データ健全性（`data_health`）④コスト（`gcp-daily-cost-report` の出力）。
