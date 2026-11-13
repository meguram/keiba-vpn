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

### 入れる順と状態（2026-11-13 夜時点。正はこの節）
**順番に意味がある**: 読み取りの権限（3）を配備（4）より先にする。先に配備すると、毎朝の点検が「一部を確かめられない」を毎日 Slack に送る。

| # | 作業 | 誰が・どこで | 状態 |
|---|---|---|---|
| 1 | VPS の SA keiba-fetch に `ops/heartbeat/vps/` だけの書き込み（`roles/storage.objectUser`、条件 `vps-sched-status`。式は `BUCKET=$BUCKET bash scripts/gcp/make_iam_condition.sh ops/heartbeat/vps/ vps-sched-status`） | 統括役（SA） | **済み**（11/13 18:4x JST） |
| 2 | ホストの cron（`sudo bash scripts/cron/install_vps_alive_cron.sh`、タグ KEIBA-VPS-ALIVE） | 統括役（root） | **済み**（11/13 18:4x JST。18:45 の回から正常） |
| 3 | SA keiba-monitor と keiba-jobs に、プロジェクトの `roles/run.viewer`・`roles/cloudscheduler.viewer`（読み取り・無料） | ユーザー（Cloud Shell。SA にはプロジェクトの IAM を変える権限が無い） | 未 |
| 4 | keiba-monitor の配備（`scripts/gcp/deploy_cloud_run_monitor.sh --apply`。live-monitor タグ付き）と Jobs の版の更新（`scripts/gcp/deploy_cloud_run_jobs.sh`、g-0730-shadow に schedule-check） | ユーザー（Cloud Shell、本番） | 未 |
| 5 | VPS の Slack の宛先: `printf '%s' '<webhook の URL>' \| sudo bash scripts/ops/vps/set_secret.sh /etc/keiba-ui/slack.env SLACK_WEBHOOK_URL`（無いと見張りの即時の通知が鳴らない。画面に「Slack の宛先が未設定」と出る） | ユーザー（VPS の root） | 未（11/13 夜、VPS に slack.env が無いことを確認） |
| 6 | d6c72260 以降を含む取得の版が VPS に届く（公開 → 毎日 03:20 の keiba-release-pull@fetch） | 昇格の流れ | 未（VPS は fetch-b316ae59＝11/12 のまま） |
| 7 | VPS の見張りのユニット: `sudo -n /opt/keiba-ui/bin/install_ops_files.sh install <版> sched-status-units`（許可リストの名前。確かめてから入れ替え、タイマーの有効化までする） | deploy（決まった操作） | 未（6 の後・ユーザーの判断） |

入れたら `config/scheduled_jobs.json` の `vps-sched-status` を `active` にし、`python -m src.scripts.docs.build_scheduled_jobs_html` で docs を作り直す（`--check` で一致を確認）。
正は 7（版のユニットを確かめてから入れ替え、タイマーを有効にする）。`scripts/ops/vps/vps_03_units.sh` も見張りのタイマーを有効にできるが（今の版に `src/ops/sched_status.py`・`config/scheduled_jobs.json` があるときだけ）、
実行した場所の手元のひな形を使い、pull のタイマーなどほかのユニットも入れ直す・有効にするので、7 とは副作用が違う。7 で入れる前に `/var/lib/keiba-fetch/sched_status` が無ければ見張りが作る（ReadWritePaths の中）。

手で試すとき: ホストの VPS 停止の検知は root の cron が状態（`logs/vps_alive_state.json`）を書くので、手では `python -m src.ops.vps_alive --dry-run` だけを使う。

### 引き継ぎ（2026-11-13）
- **済み（develop・CI 緑）**: 一覧・ルール・VPS の見張り・ポータル /schedules・毎朝の点検・ホストの VPS 停止の検知（d6c72260、de3d68ca ほか）。
  ほかに、めぐ分析の調教師×調教の表示（c17f488a、stg 1c728983 に反映済み）、調教の被覆の判定でホストが GCS を読まない修正（957858ad）。
- **裏で動いているもの**:
  - 11/07・08 の出走馬の調教の取り直し（`backfill_horse_training_recent --dates 2026-11-07 2026-11-08`。他の取得を待って 1 時間で区切られたら 5 分置いて最大 6 回まで再開）。ログ `logs/horse_training_backfill.cron.log`。
  - 調教の過去分の全ページ取得（`backfill_horse_training_past`、あと約 2 日）。終わると打ち切った馬の全ページの取り直しが自動で続く。ログ `logs/horse_training_past.log`。
- **明朝に確かめること**: 毎朝の事前計算（`export_tables_daily.sh` の中の `precompute_race_analysis`）が版 39 で作り直し、11/07 福島 1R の調教師×調教でパターンの無い馬に「集計の対象外／未取得／記録なし」が出ること。
  その後、develop の CI が緑なのを確かめて stg へまとめてマージ（`Merge develop into stg (…)`）。
- **master への昇格の前提（2026-11-13 夜、統括役の最終点検）**: 昇格（promote.yml:121 → deploy_gcp.sh jobs → scheduler_gen）が、今日削除した keiba-g-* 13 件を ENABLED で作り直す経路がある。
  2026-11-13 19:0x JST に `python -m src.ops.scheduler_gen scheduler` を表示だけで流すと、14 グループすべての create/update が出た（g-0730-shadow だけであるべき）。
  直す担当は cron-catchup-weekly-decision（job_map）。**この修正が develop・stg に入り、同じコマンドの出力が g-0730-shadow の update だけになったことを確かめてから昇格する**。
  未修正なら昇格しない（作り直されると 1 件 約 15 円/月＋Jobs の実行の費用。監視ポータルは「廃止なのに有効」と出す）。昇格はユーザーの直接の指示で。
