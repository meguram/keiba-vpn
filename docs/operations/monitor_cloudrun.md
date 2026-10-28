# 監視ポータルの Cloud Run 化（D4）

> 方針（`decoupling_plan.md` 4.6）: 運用の監視は **GCP（Cloud Run・最小 0）に置き、VPS の死活を VPS の外から見る**。ホストの `src/monitor/app.py`（Flask :23900）は研究者向けとして残してよいが、運用の監視には使わない。

## 1. 何を作ったか
- `src/monitor/cloud_app.py`（読み取り専用の Flask。ホストのローカルファイルを一切読まず、GCS の `ops/` の JSON だけを見る）。
  - VPS ハートビート（`ops/heartbeat/latest.json`。`heartbeat_schema.evaluate` で異常を出す）
  - Jobs の最新の実行（`ops/job_runs/<job>/latest.json`。状態・所要・メモリのピーク/上限・並走か）
  - 今日の異常通知（`ops/alerts/<日>/`）
  - 画面・取得役の配布（`ops/releases/<kind>/latest.json` と VPS が報告する `applied/vps.json`。版ずれが一目で分かる）
  - 成果物の一致（`ops/parity/<日>/`）と切替判定（`ops/cutover/status.json`）
  - （2026-10-28 に画面を拡張。下の 5 章）`/ops` が従来の表（全体の信号つき）、`/api/overview` が同じ内容の JSON、`/api/health` が起動確認（鍵なし。※ `/healthz` は Cloud Run の前段(Google Frontend)が横取りして HTML の 404 を返すので使わない）。
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
4. 起動確認: `curl -s $(gcloud run services describe keiba-monitor --region asia-northeast1 --format='value(status.url)')/api/health`（`{"status":"ok"}`）。鍵を付けて `/api/overview` が返ること。
5. 公開の道（VPS と Cloudflare。ここは VPS 側の担当）: `monitor.megu-keiba.com` の向き先を、ホストの :23900 から「VPS の nginx → keiba-monitor の Cloud Run URL（`X-Keiba-Proxy-Key` を付与）」に切り替える。Cloudflare Access の保護は今のまま。切り替えが済んでから、ホスト側の Tunnel のルートを外す。

## 3. 費用
無料枠の範囲の見込み（`decoupling_plan.md` 7 章。リクエストは人が見るときだけ、最小 0、最大 1 台）。読み取りは GCS の小さな JSON が 1 表示あたり数十件（Class B）。

## 4. 未対応（ホストの監視ポータルにあって、ここには無いもの）
ホストの systemd/プロセス・git の状態・ログの tail・スクレイピングのカバレッジ表は、ホスト（や VPS）のローカルを見るため移していない。必要なものは、収集側が GCS に JSON を出す形（ハートビートと同じ）にしてから足す。優先度の高い順: ①取得役（VPS）のタスクの完了/失敗（`ops/fetch_tasks/`）②予測成績（`ops/prediction_results/`）③データ健全性（`data_health`）④コスト（`gcp-daily-cost-report` の出力）。

## 5. 画面の拡張とVPS からの受信（2026-10-28）
**ホスト（研究用 PC）とは一切つながない**。材料は GCS と、VPS からの署名つき POST、Cloud Run への HTTP 確認だけ。

| 画面 | 内容 | 材料 |
|---|---|---|
| `/`（ホーム） | stg / prod × VPS / GCP の 4 マス、直近の開催日パイプライン、アクセス、SEO、今日の異常 | ハートビート、`ops/job_runs`、API の `/api/health`（URL は環境変数 `KEIBA_API_URL_PROD` / `KEIBA_API_URL_STG`）、下の各画面 |
| `/pipeline` | 直近の開催日ごとに 出馬表→オッズ→予測→速報→指数→確定→予測成績 が最後まで終わったか（レース別も） | GCS の `chuou/data/preprocessed/netkeiba/pc/…`・`state/data/local/inference/…`・`ops/prediction_results/` |
| `/traffic` | 訪問者・ページビュー・推移 90 日・上位ページ 10・流入元・端末・時間帯・クローラー・404/5xx | VPS が送る `ops/traffic/bundle.json` |
| `/seo` | SEO スコア・技術/コンテンツ/公開前の残り作業・ページ別の指摘 | CI が置く `ops/seo/audit.json` |
| `/ops` | 従来の Jobs・配布・成果物の一致の表 | 従来どおり |

**受け先を API ではなくここ（keiba-monitor）にした理由**: `keiba-api` は『CPU を常に割り当て』（インスタンス課金）で、15 分以内の定期リクエストがあると落ちずに月 約 7,800 円になる（`cost_recalc_20261028.md`）。monitor はリクエスト課金・最大 1 台。
このため VPS のハートビートからも `/api/` の確認（`api_upstream_http`。nginx 経由で Cloud Run に届く）を外し、`nginx_http`（`/version.json`）に変えた。stg の画面（`/srv/keiba-ui-stg` がある場合）は `stg_ui_http` と `keiba-ui-stg` を追加で見る。
ホームの API 確認は人が画面を開いたときだけ・10 分使い回し（`?probe=0` で省略）。

受信（署名は `KEIBA_HEARTBEAT_SECRET`。共有ヘッダ鍵は不要）: `POST /ops/heartbeat`（5 分おき）、`POST /ops/traffic`（15 分おき）。`POST /ops/heartbeat/check`（Scheduler が 15 分おき。共有ヘッダ鍵が要る）で異常・復旧を Slack に通知。

### ユーザーの作業（GCP / VPS / CI）
1. **SA keiba-monitor の権限を足す**（今は `ops/` の読み取りだけ）: ①書き込み `ops/heartbeat/` と `ops/traffic/`（`roles/storage.objectUser` を `make_iam_condition.sh` で prefix 限定）。②読み取り・一覧 `chuou/data/preprocessed/netkeiba/pc/` と `state/data/local/inference/`（開催日パイプライン用）。
2. 秘密 `KEIBA_HEARTBEAT_KEY` と `SLACK_WEBHOOK_URL` を SA に読ませる（`roles/secretmanager.secretAccessor`）。
3. 配備し直す: `KEIBA_API_URL_PROD=… KEIBA_API_URL_STG=https://rstg---…run.app bash scripts/gcp/deploy_cloud_run_monitor.sh --apply`（リクエスト課金のまま。`--cpu-throttling` を明示）。
4. Scheduler: `POST <monitor>/ops/heartbeat/check`（15 分おき、`X-Keiba-Proxy-Key` 付き）。API 側（`/ops/heartbeat/check`）に向けた同種のジョブを作っていれば**止める**（API が 15 分おきに起きて課金されるため）。
5. VPS: `/etc/keiba-ui/heartbeat.env` の `HEARTBEAT_URL` を monitor の `…/ops/heartbeat` に変え、`sudo bash scripts/ops/vps/vps_07_traffic_timer.sh`（アクセス集計の送信タイマー。root で nginx のログを読む）。ハートビートの送信側（`heartbeat_client.py`）も取り込む。
6. SEO: CI の昇格後に `python -m src.scripts.ops.seo_audit_publish` を実行する 1 ステップを足す（`promote.yml` / `deploy-stg.yml` の GCS 認証済みジョブ。`ops/seo/` への書き込み権限が要る）。それまでは SEO 画面は『まだ点検結果がありません』。
7. 公開の道は 2 章 5 のとおり（`monitor.megu-keiba.com` の向き先をホストの :23900 から Cloud Run の monitor に）。切り替えるまでは今の URL にこの画面は出ない。
