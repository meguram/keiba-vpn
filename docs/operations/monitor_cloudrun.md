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
| ホームの『API の課金ガード』/ `/api/cost` | keiba-api の課金される起動時間（直近24時間の平均台数・月額の見込み・夜間に起動していた時間）。異常は Slack に通知し、ホーム全体の信号にも反映 | Cloud Monitoring `run.googleapis.com/container/billable_instance_time`（`src/ops/api_cost_guard.py`） |

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
8. **API の課金ガード**: monitor の SA に `roles/monitoring.viewer`（プロジェクト）を付ける。付けるまでホームは『確認できない（権限がありません）』と出るだけで、ほかは動く。点検は手順 4 の Scheduler（`/ops/heartbeat/check`）が一緒に行う（ジョブを増やさない）。
   - 判定: 夜間（JST 1〜4 時台）に 30 分以上起動した時間が 2 時間分以上 → 警告（4 時間分以上で重大）／ 直近 24 時間から月額の見込みが 800 円（API 分の目安）以上 → 警告（1,600 円以上で重大）。無料枠の数値は未確認なので、画面に『無料枠なし』の値も並べる。
   - 背景: ping の発信元を列挙しても漏れる（2026-10-28 は VPS のヘルスチェック、ホストの cron の 2 つで、1〜2 台が 24 時間起動していた）。原因ではなく結果（課金される起動時間）を見張る。

## 定期実行ジョブの見張り（/schedules、2026-11-13）

一覧の正は `config/scheduled_jobs.json`（docs のページ `docs/html/operations/final/scheduled_jobs.html` も `python -m src.scripts.docs.build_scheduled_jobs_html` でここから作る）。
判定のルールは `src/ops/schedule_rules.py` の 1 か所で、次の 3 か所が同じ関数を使う。**Cloud Run を起こす定期の処理は足していない**。

| 役 | どこで | 何を読む・書く |
|---|---|---|
| VPS の見張り `keiba-sched-status.timer`（15 分おき） | VPS（`python -m src.ops.sched_status`） | `systemctl show` を読み、GCS の `ops/heartbeat/vps/status.json` に 1 回書く。赤の発生・解消だけ Slack（`/etc/keiba-ui/slack.env`） |
| ポータル `/schedules`・ホームのカード | Cloud Run keiba-monitor（開いたときだけ・5 分使い回し・`Cache-Control: private, max-age=300`） | `ops/heartbeat/vps/status.json`・`ops/job_runs/<job>/latest.json`・Scheduler と Run Admin の API（読み取り） |
| 毎朝の点検 `schedule-check` | GCP の g-0730-shadow（07:30）の中 | 同上。赤があれば Slack。VPS の記録が 45 分途絶していれば「VPS の記録の途絶」 |
| VPS 全体の停止の検知（主） | ホストの cron 15 分おき（`install_vps_alive_cron.sh`、タグ KEIBA-VPS-ALIVE） | `https://app.megu-keiba.com/version.json`（Cloudflare → VPS。`/api/` ではない）。2 回続けて届かなければ Slack・戻ったら Slack |

### 月額の見積もり（最悪の回数。2026-11-13）
- VPS → `ops/heartbeat/vps/status.json`: Class A 1 回 × 96 回/日 × 30 = 2,880 回 ≒ **2.3 円**（VPS → GCS の書き込みは転送料 0）
- ポータルを開く: Class B 約 3 回＋無料の API 6 回 × 最悪 96 回/日 × 30 ≒ 0.5 円。keiba-monitor の起動し直し（512Mi・最小 0）は最悪 CPU 約 3 秒 × 2,880 回 ≒ 31 円（無料枠の中なら 0 円）
- 毎朝の点検: 既存のジョブに 1 手順（GCS 数回・数秒）≒ 0 円。ホストの cron の確認: GCP・GCS に触れない 0 円
- 合計 **ふだん 約 3 円、最悪 約 35 円/月**。新しい Scheduler・常駐・最小インスタンス・15 分未満の ping は無い

### ユーザーの作業（権限と配備。統括役経由で確認してから）
1. **VPS の SA keiba-fetch に `ops/heartbeat/vps/` だけの書き込み（中身は status.json の 1 件）**:
   `BUCKET=$BUCKET bash scripts/gcp/make_iam_condition.sh ops/heartbeat/vps/ vps-sched-status > /tmp/cond_vs.yaml && gcloud storage buckets add-iam-policy-binding gs://$BUCKET --member "serviceAccount:keiba-fetch@$PROJECT.iam.gserviceaccount.com" --role roles/storage.objectUser --condition-from-file /tmp/cond_vs.yaml`
2. **読み取りの権限**（どれも無料の読み取り）: SA keiba-monitor と keiba-jobs に、プロジェクトの `roles/run.viewer` と `roles/cloudscheduler.viewer`。
   付けるまでは GCP の欄が「GCP の状態を読めない（権限がありません）」の 1 件になるだけで、画面は落ちない。
3. 配備: monitor を配備し直す（`scripts/gcp/deploy_cloud_run_monitor.sh --apply`）。Jobs の版を更新（`scripts/gcp/deploy_cloud_run_jobs.sh`。g-0730-shadow に schedule-check が入る）。
   VPS は fetch の版（`scheduled_jobs.json` と `src/ops/sched_status.py` を含む）が入った後に `sudo bash scripts/ops/vps/vps_03_units.sh`（見張りのタイマーを有効化）。
   ホストは `sudo bash scripts/cron/install_vps_alive_cron.sh`。入れたら `config/scheduled_jobs.json` の `vps-sched-status`・`host-vps-alive` を `active` にする。

### 引き継ぎ（2026-11-13 18:00 締め）
- **済み（develop・CI 緑）**: 一覧・ルール・VPS の見張り・ポータル /schedules・毎朝の点検・ホストの VPS 停止の検知（d6c72260、de3d68ca）。
  ほかに、めぐ分析の調教師×調教の表示（c17f488a、stg 1c728983 に反映済み）、調教の被覆の判定でホストが GCS を読まない修正（957858ad）。
- **まだ配備していない**（明日。ユーザーの確認後）: stg への反映（d6c72260 以降）、IAM（上の 1・2）、monitor・Jobs の配備、VPS の `vps_03_units.sh`、ホストの `install_vps_alive_cron.sh`。
  入れたら `config/scheduled_jobs.json` の `vps-sched-status`・`host-vps-alive` を `active` にし、`python -m src.scripts.docs.build_scheduled_jobs_html` で docs を作り直す。
- **裏で動いているもの**:
  - 11/07・08 の出走馬の調教の取り直し（`backfill_horse_training_recent --dates 2026-11-07 2026-11-08`、323 頭・30〜60 分。他の取得を待って 1 時間で区切られたら 5 分置いて最大 6 回まで再開）。ログ `logs/horse_training_backfill.cron.log`。
  - 調教の過去分の全ページ取得（`backfill_horse_training_past`、12,837 / 19,119 頭、あと約 2 日）。終わると打ち切った馬の全ページの取り直しが自動で続く。ログ `logs/horse_training_past.log`。
- **明朝に確かめること**: 毎朝の事前計算（`export_tables_daily.sh` の中の `precompute_race_analysis`）が版 39 で作り直し、11/07 福島 1R の調教師×調教でパターンの無い馬に「集計の対象外／未取得／記録なし」が出ること。
