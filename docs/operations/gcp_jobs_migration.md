# ジョブ・cron の GCP 化と、ホストの cron 停止（フェーズ 3〜5）

設計の根拠は `architecture_target.md`、進める順序の一枚は `migration_runbook.md`、GCP 側の初期設定は `gcp_setup_runbook.md`、VPS の取得の実行役は `vps_agent.md`。
この文書は **「どのジョブを・どの順で・どう並走して・どう切り替えて・どう戻すか」** の正本です。コマンドと判定は全部リポジトリにあり、実行はあなた（Cloud Shell / ホスト）が行います。

**凡例**: 「ユーザー」= あなたの作業。dev PC と私は GCP・ホストに接続しないので、`gcloud` と root の操作は必ずあなたが実行します（私は手順・スクリプト・判定ツールを用意するところまで）。
**共通のルール**: 秘密は画面・リポジトリに出さない／ホストへは git pull せずファイル単位／本番（prod）への昇格は毎回手動／各段階で旧経路をすぐ戻せる状態を保つ／**2026 年のデータは学習・選択に使わない**（ここでは運用の移行だけで、モデルや特徴量の中身は変えない）。

---

## 0. 用意したもの（一覧）

| 役割 | 場所 | 内容 |
|---|---|---|
| **ジョブ対応表（正本）** | `config/gcp_job_map.json` | ホストの全 cron（実機の root crontab を 2026-10-21 に読み取り）を `gcp` / `vps` / `host` / `retire` に分類。ジョブごとに元の cron（UTC）・移行後の時刻（JST）・コマンド・メモリ・環境変数名・入出力（state）・使うモデル・並走検証の方法・準備状況（ready と blockers）・切替判定の基準。検証: `python -m src.ops.job_map validate`（テスト `tests/ops/test_gcp_job_map.py`） |
| ディスパッチャ | `src/ops/job_dispatch.py` | Cloud Run Jobs の**唯一の入口**。対応表の `gcp` ジョブだけを実行（許可リスト）。メモリ予算・準備未完了・依存・ガード（開催日／取得完了）・タイムアウト・実行記録・失敗通知 |
| 入出力の同期 | `src/ops/state_sync.py` | `state/`（GCS）⇄ コンテナ。合計サイズに上限（ディスクはメモリ上のため）。ホストで `plan` を実行すると実サイズが分かる |
| モデル配布 | `src/ops/model_publish.py` | ホストが `publish`（sha256 の manifest → 最後に `latest.json`）、ジョブが検証つきで `fetch`。壊れていたら何も置かない |
| 予約（司令塔） | `src/ops/reservations.py` | 前夜: 全レースの取得タスク（T-60/T-15/直前/馬体重の検知、`fetch_plan.plan_day` を再利用）と、予測の Cloud Tasks 予約（Run Job を起動）。毎時: VPS 実行役の定期タスクを cron 式から展開 |
| イベント | `src/ops/events.py` | VPS 実行役が置く `ops/events/weights_announced/…` を見つけて即時の予測タスクを積む（`event-pump`） |
| 前夜の分離 | `src/scraper/auto_scrape.py`（`raceday-eve-fetch`）・`src/ops/jobs/eve_precompute.py` | `raceday-eve` を「取得（VPS）」と「計算（GCP）」に分けた。計算は従来と同じ `_eve_precomputes` |
| 予測ジョブ | `src/ops/jobs/predict_race.py` | 1 レースの特徴量づくり＋予測（従来の `run_for_races`。VPS は ML を持たない） |
| 監視 | `src/ops/vps_watchdog.py` | VPS のハートビート（15 分途絶=緊急、MemAvailable<1.0GB×3 回=注意、<600MB=緊急、swap>200MB、ユニット failed／再起動多発、ディスク>80%）＋取得タスクの滞留・失敗＋予約の鮮度 |
| コマンド生成 | `src/ops/scheduler_gen.py`・`scripts/gcp/deploy_cloud_run_jobs.sh`・`Dockerfile.jobs` | Cloud Run Jobs（3 つ）・Scheduler（グループごと）・Cloud Tasks キューの `gcloud` コマンドを対応表から生成。既定は表示のみ、`--dry-run` あり |
| 並走検証 | `src/scripts/ops/parity_check.py` | ホストと GCP の成果物（parquet／JSON／ファイル）を許容誤差つきで比較、`ops/parity/` に記録、Slack に要約 |
| 切替判定 | `src/scripts/ops/cron_cutover_status.py` | ジョブごとの合否表（機械判定）。同じ cron_job_id のジョブが全部合格したときだけ「止めてよい」 |
| 切替の適用 | `src/ops/job_owner.py`・`config/job_owner.json`・`scripts/cron/cron_run.sh`・`scripts/ops/cutover/host_cron_switch.sh`・`src/scripts/ops/cutover_plan.py` | 担当の切替（ホストの cron が何もしなくなる）→ crontab の行のコメントアウト。バックアップ・再有効化つき |
| 並走用ホスト cron | `scripts/cron/install_host_snapshot_cron.sh` | 並走期間だけ、ホストの成果物を `host_snapshot/<日付>/` に上げる |

---

## 1. 全体像

```
Cloud Scheduler（グループごと 1 本・全 13 グループ）──▶ Cloud Run Job（s/m/l の 3 つ。同じイメージ）
   args: -m src.ops.job_dispatch group <g-…>                       │
Cloud Tasks（keiba-predict）── 前夜に予約 ──▶ 同じ Run Job（l）      ▼
   args: … job predict-race --arg race_id=… --arg date=…   src.ops.job_dispatch
                                                                   ├ ガード（開催日・取得完了・メモリ・準備状況）
VPS 実行役 ─ weights_announced ─▶ GCS ─▶ event-pump ─▶ Cloud Tasks   ├ models/ を検証つきで取得（sha256）
   ▲  pull（60 秒）                                                 ├ state_in を pull（上限あり）→ steps（直列）→ state_out を push
   └─ ops/fetch_tasks/pending ◀── reserve-race-day / reserve-fetch-tasks（GCP が積む）
記録: ops/job_runs/・ops/alerts/・ops/parity/・ops/cutover/status.json（すべて GCS）
```
- **Scheduler の本数は 13**（無料枠 3 本を超える 10 本 × 約 15 円 ≒ ¥150/月。上限 14）。類似のジョブは 1 グループにまとめ、グループ内は直列（メモリを増やさない）。
- **Run Job は 3 つ**（s=512Mi／m=2Gi／l=4Gi）。グループの最大メモリに合わせて選ぶ。ジョブの宣言メモリがコンテナの上限を超えるときは実行せず `refused`（緊急通知）。
- 取得（netkeiba）は VPS の実行役だけ。GCP のイメージにブラウザ・取得の秘密は入れない。

### 通知で「どちらの環境か」を見分ける（並走期間）
`src/utils/slack_notify.send_payload` が `KEIBA_ENV`／`KEIBA_RUN_LOCATION` から【STG】【PROD】と末尾の「環境・実行場所」を自動で付けます（新しいコードは Slack を直接叩かず、必ずこの関数を通る。テストで検査）。
Run Job の環境変数には**必ず** `KEIBA_ENV=prod`・`KEIBA_RUN_LOCATION=gcp` が入る（対応表の `runtime.env`、`scheduler_gen` が出力、`validate` が検査）。

| 送り元 | 見え方 |
|---|---|
| ホスト（並走中は stg のまま） | 【STG】… ／ 環境: STG ・ 実行場所: ホストPC |
| GCP の Run Job | 【PROD】… ／ 環境: PROD ・ 実行場所: GCP |
- 宛先を分けたいときは `SLACK_WEBHOOK_URL_PROD`／`_STG`（`slack_notify.webhook_url` が対応）。分けなければ同じチャンネルに 2 系統の通知が並ぶ（見分けは上の表）。
- 並走中に同じ内容の通知が 2 通届くのは想定どおり（片方は検証用）。**切替後は PROD だけ**になる。

---

## 2. ジョブの分類（全体は `python -m src.ops.job_map list`）

| 移行先 | ジョブ（cron_job_id／ジョブ名） | 備考 |
|---|---|---|
| **GCP（Run Jobs）** | 騎手・調教師統計／履歴テーブル（export_tables_daily）／サービング状態（precompute・profile）／履歴特徴量（feature-accumulate）／ラップ予想（朝・前夜）／前夜の計算（eve-precompute）／追走難度 v2 の週次突き合わせ／ドリフト監視・凍結追跡（月曜）／オッズ乖離（日曜）／馬名インデックス（金曜）／Slack 通知（結果・異常・サマリー・網羅）／ページ品質／コスト報告／**新規**: 予測（predict-race）・予約（reserve-race-day・reserve-fetch-tasks）・event-pump・vps-watchdog・parity-check | Scheduler 13 本。`ready=false` のものは理由（blockers）を解消するまで動かない |
| **VPS 実行役**（取得タスク） | entries-sync／daily-race-lists（朝・夕）／jra-baba-morning／recent-retry（朝・夕）／barometer-retry／各レースの T-60・T-15・直前オッズ（raceday-runner の置き換え）／各レースの馬体重の検知（raceday-weight-runner の置き換え）／raceday-evening／raceday-comment／new-horse-pedigree／weekly-update／raceday-eve-fetch | 実行役が既定で担当するのは `fetch_task.DEFAULT_AGENT_TASKS` のみ。重いものは **VPS の 250M で実測してから** `config/fetch_owner.json` の `agent_tasks` に足す（段階移行）。担当外のタスクは**積まない**（積むと誰も拾わず期限切れの failed が溜まるため） |
| **ホストに残す**（停止対象外） | 長期バックフィル（backfill-*）／調教の取り直し（horse-training-backfill）／馬の備考（horse-remarks-daily）／誕生日データ（sporthorse-births-nightly）／速報結果ランナー（raceday-result-runner。常駐ループで許可リストに無い）／メンテナンス自動反映（stg 用）／遠隔アクセス／監視ポータルの死活（フェーズ 5 まで）／ログローテーション／server_watchdog／**モデリング全般** | `job_owner.json` に載せられない（テストで検出）。`cutover_plan` も作らない |
| **廃止** | sync_pg_from_gcs／ensure_postgres | PostgreSQL は廃止（API が Cloud Run に移ってから止める） |

`raceday-eve` は 1 本の cron を **取得（`raceday-eve-fetch`＝VPS）** と **計算（`eve-precompute`＝GCP）** に分けました。取得は `KEIBA_EVE_FETCH_ONLY=1` で計算をスキップ、計算は 19:10／19:40 JST に 2 回走らせ（冪等）、取得完了を `ops/fetch_tasks/done/raceday-eve-fetch-<日>…` で確認してから動きます（20:15 になっても未完了なら緊急通知）。

---

### 2.1 「ホストに残す」の見直し（2026-10-22）
最終形（公開はホストと無関係。ホストは研究専用）の観点で、host 21 件＋retire 2 件を見直した（`config/decoupling_roles.json`、`tests/ops/test_decoupling_roles.py`）。**研究専用は 5 件だけ**（`modeling`・`sporthorse-births-nightly`・`backfill-2020/2021-fast`・`keiba-remote-access`）。**運用に必要なのに host だった 12 件**（`raceday-result-runner`・`horse-remarks-daily`・`backfill-2022〜2026`・`backfill-all-horse`・`horse-training-backfill` の当年分）は VPS/GCP へ移す。廃止・置き換えは 6 件。理由と移行順序は [decoupling_plan.md](./decoupling_plan.md) の 3 章・5 章。`destination` はこの時点では変えていない（移すジョブから順に job_map を更新する）。

## 3. データとモデルの持ち込み

### 3.1 入出力（state）
- コンテナは毎回空で、書き込みはメモリを使います。各ジョブの `state_in`／`state_out` に**サイズ上限（max_mb）**があり、超えると取らずに失敗します。`state_in の合計 + 600MB ≤ memory_mib` も `validate` で検査します。
- **最初に**ホストで実サイズを確認します（上限の見直し材料）:
  ```bash
  cd /home/jovyan/work/keiba-vpn
  for j in jockey-trainer-stats export-tables-daily serving-state-precompute lap-forecast-morning drift-monitor-weekly; do
    PYTHONPATH=. python3 -m src.ops.state_sync plan --job $j
  done
  ```
- 初回の種まき（ホスト → GCS。アップロードは無料）: `PYTHONPATH=. python3 -m src.ops.state_sync push --job <ジョブ> --which in` （`--which in` で state_in の中身を `state/` に上げる）。以後は GCP のジョブが自分で更新します（`state/` は GCP 専用で、ホストの本番データには触れません）。
- VPS 実行役の結果のうち**ローカルだけにあるもの**（`race_lists`・`race_day_schedule`・`cushion`）は、タスク完了時に `state/` へ上げる必要があります（`reserve-race-day` がこれを読む。対応表の VPS ジョブの `state_out` に記載）。これは vps-agent 側の作業で、未了の間は `reserve-race-day` が `ready=false` です。

### 3.2 モデル（ホスト → GCS。ジョブは検証つきで取得）
```bash
# ホストで（モデルを更新するたび。同じ内容は再アップロードしない）
for c in bundle_v4 stage_v4 aux_v4 bundle_v5 tl_reference_v5 serving_bundle lap_models final_odds; do
  PYTHONPATH=. python3 -m src.ops.model_publish publish --component $c
done
PYTHONPATH=. python3 -m src.ops.model_publish status
```
- `local_path` はホストの実際の場所に合わせて `config/gcp_job_map.json` の `models` を直す（既定は `v3_serving.py` の既定パスから推定。**`status` で `published: true` になるまで確認**）。
- 取得は `models/<component>/latest.json` → `manifest.json`（sha256）→ 各ファイルの size／sha256 の順に検証し、失敗したら何も置きません（古い版が残る）。環境変数（`KEIBA_V5_BUNDLE` など）は取得先に自動で向きます。想定オッズのモデルのようにコードが固定パス（`models/final_odds_bundle.json`）を読むものは、取得後にそこへシンボリックリンクを張ります（`install_to`）。
- 学習の成果物を GCS → ホストへ戻すことはしません（外向き転送が有料。`architecture_target.md` 5.3）。

---

## 4. 移行の順序

> 原則: **低リスク（通知・集計・監視）→ 予測生成 → 取得**。1 ジョブ（cron_job_id の単位）ずつ、「並走 → 判定 → 担当の切替 → 観察 → 行の停止」。

### 4.0 事前（ユーザー。1 回だけ）
1. `gcp_setup_runbook.md` の 1〜8 章（プロジェクト・サービスアカウント `keiba-jobs`／`keiba-scheduler`・秘密・GCS 監査・予算アラート・課金エクスポート）。
2. 権限を 2 つ足す（Scheduler／Tasks が Run Job を起動し、`keiba-jobs` がその身元でタスクを積むため）:
   ```bash
   PROJECT=$(gcloud config get-value project)
   # Run Job の起動（run.invoker に run.jobs.run が含まれるか gcp_setup_runbook.md 3 章で確認した結果に従う。含まれなければ Job ごとに）
   for j in keiba-dispatch-s keiba-dispatch-m keiba-dispatch-l; do
     gcloud run jobs add-iam-policy-binding $j --region asia-northeast1 --member "serviceAccount:keiba-scheduler@$PROJECT.iam.gserviceaccount.com" --role roles/run.developer
   done
   # keiba-jobs が keiba-scheduler の身元でタスクを積める（act-as）
   gcloud iam service-accounts add-iam-policy-binding keiba-scheduler@$PROJECT.iam.gserviceaccount.com \
     --member "serviceAccount:keiba-jobs@$PROJECT.iam.gserviceaccount.com" --role roles/iam.serviceAccountUser
   ```
   （Run Job を作る前は `add-iam-policy-binding` が失敗します。4.1 の「Run Job を作る」の後に実行）
3. イメージ: `bash scripts/gcp/deploy_cloud_run_jobs.sh build` の手順（Cloud Build。無料枠内）。
4. モデルの公開: 3.2。
5. ホストへ必要なファイルを反映（ステージ → 03:05 の窓）: `src/ops/`・`src/scripts/ops/`・`scripts/cron/cron_run.sh`・`scripts/cron/install_host_snapshot_cron.sh`・`scripts/ops/cutover/`・`config/job_owner.json`・`config/gcp_job_map.json`。**`cron_run.sh` はホストの全 cron 行が通る**ので、反映後に `config/job_owner.json` が空（`owners: {}`）のままで従来どおり動くことを確認（空の間は何も変わらない。壊れていても・python が無くても従来どおり実行する作り）。

### 4.1 Run Job・キュー・Scheduler を作る（ユーザー。Cloud Shell）
```bash
export PROJECT_ID=$(gcloud config get-value project) GCS_BUCKET=<既存のバケット名> REGION=asia-northeast1
cd ~/keiba-vpn
bash scripts/gcp/deploy_cloud_run_jobs.sh show all --parallel            # 中身を確認（実行しない）
bash scripts/gcp/deploy_cloud_run_jobs.sh apply queue                    # Cloud Tasks キュー keiba-predict
bash scripts/gcp/deploy_cloud_run_jobs.sh apply jobs --parallel          # Run Job 3 つ（並走期間は --parallel = KEIBA_PARALLEL_RUN=1）
# → 4.0 の 2 を実行
bash scripts/gcp/deploy_cloud_run_jobs.sh run-now group g-pump --dry-run # 手動で 1 グループ試す前の確認
```
- `--parallel` は予測を**影モード**に固定します（`predict-race` の `parallel_env`: `KEIBA_V3_SERVING=shadow`・`KEIBA_V5_SHADOW=1`）。GCP の予測が本番の `race_predictions` を上書きしません。
- **Scheduler は「準備できたジョブ」があるグループだけ**作られます（`apply scheduler`）。準備の完了は、blockers を解消してから `python -m src.ops.job_map ready <ジョブ> --note "確認した内容"` で記録し、再度 `apply scheduler` します（作成は冪等）。最初に動くのは `reserve-fetch-tasks`／`event-pump`／`odds-drift-weekly` で、他は blockers の解消待ちです。

### 4.2 着手順と、ジョブごとの blockers の解消方法

| 順 | ジョブ（cut_unit） | 主な blockers と解消方法 |
|---|---|---|
| 1 | `vps-watchdog` | VPS のハートビート送信タイマーを有効にしてから（それまでは「一度も届かない」緊急通知になる）。`vps_ui_deploy.md`／`vps_agent.md` の手順 |
| 1 | `slack-scrape-*`・`slack-results` | 元データ（`data/local/meta/scrape_*`・`prediction_results`）の置き場。取得状況は VPS 実行役の `ops/fetch_tasks/`、予測成績は `state/` から読む形への切替が必要 → 別途小改修（未了。後述の未解決） |
| 1 | `gcp-daily-cost-report` | 課金エクスポート有効化＋`GCP_BILLING_BQ_TABLE`（`gcp_setup_runbook.md` 8 章） |
| 2 | `odds-drift-weekly` | 準備済み。月曜の窓で並走 → 一致したら切替 |
| 2 | `jockey-trainer-stats`・`export-tables-daily`・サービング状態系 | 4.3 の plan で state の実サイズを確認 → 種まき → 並走 |
| 2 | `lap-forecast-*`・`eve-precompute`・`predict-race` | モデルの publish（3.2）／`eve-precompute` は取得の VPS 移行（`raceday-eve-fetch` が `agent_tasks` に入る）か GCS に出馬表が揃っていること |
| 2 | `drift-monitor-weekly`・`frozen-tracking-weekly`・`tracking-v2-weekly` | full エンジンは全履歴を読む（4Gi に収まるか並走で実測）。bundle エンジンへの切替を検討 |
| 3 | 取得系（VPS 実行役） | `vps_agent.md` の段階移行（`agent_tasks` を 1 つずつ足す）。足すと同時に `cutover_plan` の「担当の切替」と同じ効果（ホストの `auto_scrape.sh` が実行しなくなる） |

> ready の更新は `config/gcp_job_map.json` を直接直しても同じです（`python -m src.ops.job_map validate` で整合を確認）。blockers を空にしたら `ready: true`。

### 4.3 並走（ユーザー。数日〜1 開催週）
並走中は**ホスト（stg）が本番相当のまま**動き、GCP は検証用に同じ処理を行います（予測は影モード、`state/` は GCP 専用）。

1. ホストに成果物の公開 cron を入れる（並走の間だけ）:
   ```bash
   sudo bash scripts/cron/install_host_snapshot_cron.sh /home/jovyan/work/keiba-vpn      # 07:50 JST に host_snapshot/<日>/ へ。外すときは --remove
   ```
2. 日々の確認（Cloud Shell かホスト）:
   ```bash
   python -m src.scripts.ops.cron_cutover_status                 # 全ジョブの合否表
   python -m src.scripts.ops.parity_check auto --date YYYYMMDD --write --send   # 手動で一致判定（parity-check ジョブが毎朝 08:00 に実行）
   ```
3. 一致しない場合の見方: `ops/parity/<日>/<ジョブ>.json` の `problems`（キーの差・列の差・最大差）。許容誤差は対応表の `parity.rtol/atol`。生成時刻などの揮発キーは JSON 比較から除外済み。
4. 開催日の確認: T-60／T-15 の予測が間に合うか（`ops/job_runs/predict-race/` の開始時刻とレースの発走時刻）。VPS 実行役の取得は `race-odds-*` の done／failed。

### 4.4 切替の基準（機械判定。`cron_cutover_status`）
| 基準 | 内容（既定。ジョブごとに `cutover` で上書き） |
|---|---|
| 直近 N 回の成功 | `min_consecutive_success`＝5 回連続（週次ジョブは 2、開催日のみのジョブは 4）。スキップ・待機は数えない。件数が足りなければ「データ不足」で不合格 |
| 実行時刻の遅れ | 予定からの開始の遅れが GCP は 600 秒以内、VPS のタスクは claim−not_before が 300 秒以内（entries-sync 等は 180 秒） |
| 成果物の一致 | `parity.kind ≠ none` のジョブは直近 3 回（週次は 2 回）の並走検証がすべて一致 |
| 異常通知ゼロ | 直近 14 日（週次は 35 日）に、そのジョブの警告以上の通知（`ops/alerts/`）が 0 件 |
| 開催日の取得 | `race-odds-per-race`: 直近 3 回の開催日で**全レース**の T-60／T-15 が成功し、失敗が 0（レース数は `race_day_schedule` の枠数と照合） |
| 影モード（`shadow_days`） | **VPS の取得ジョブのみ**（`require_shadow`＝既定 true）。ホストが取得を続けたまま VPS が並走した日次レポートが、直近 5 日（開催日を含む）すべて合格。足りない・読めないときは不合格。対象外のタスク（重い定期処理）は人が `shadow.exempt`（理由つき）を書くまで止められない。詳細: `vps_agent.md` 5.5 章 |
| 対象外 | `host`（ホストに残す）は「N/A」、`ready=false` は「未着手」＋理由、`retire` は「対象外」 |
- **VPS の取得を含む単位は、影モードの基準を満たすまで、担当の移譲（`job_owner set`）も `cutover_plan make`（`--force` でも）も `host_cron_switch.sh disable`（`can-stop`、終了コード 5/6）も拒否される。** 「prod で数日間データ取得が正常にクリアするのを見届けるまで、stg（ホスト）の取得は続ける」をコードで保証している。
- 同じ `cron_job_id` を持つジョブ（例: 朝・夕の `daily-race-lists`、`raceday-runner`＝予約(GCP)＋オッズ(VPS)）は**全部合格したときだけ**止めてよい。
- 結果は `--json`／`--markdown`／`--write`（`ops/cutover/status.json`）で出せます。実行は読み取りのみです。

### 4.5 切替と、ホストの cron の停止（ユーザー。1 つずつ）
**二重実行を避ける順序**: ① 判定が合格 → ② **担当を渡す**（ホストの cron は何もしなくなる。crontab の行は残る）→ ③ 1〜2 週間観察 → ④ **crontab の行をコメントアウト**。②の瞬間も、GCP とホストが同じ処理を取らない（ホストは実行前に担当を確認して見送る）。

```bash
# ホスト（stg）で。判定結果を作る（GCS を読むのでホストか Cloud Shell で）
cd /home/jovyan/work/keiba-vpn
PYTHONPATH=. python3 -m src.scripts.ops.cron_cutover_status --json > /tmp/cutover_status.json
PYTHONPATH=. python3 -m src.scripts.ops.cutover_plan list                       # 止める単位(cron_job_id)と波(wave1=通知・監視 → 2=予測・集計 → 3=取得)
# 例: drift-monitor-weekly（合格していない単位は作らない。--force は検証用）
PYTHONPATH=. python3 -m src.scripts.ops.cutover_plan make --unit drift-monitor-weekly --status-file /tmp/cutover_status.json --out-dir /tmp
bash /tmp/cutover_drift-monitor-weekly_1_owner.sh          # ② 担当を GCP に（config/job_owner.json）。VPS の取得は fetch_owner.json の agent_tasks にも足す
# …観察（毎日: cron_cutover_status が合格のまま・Slack に異常通知が無い。cron_runs の記録は「GCP の担当」として成功で残る）…
sudo bash /tmp/cutover_drift-monitor-weekly_2_cron_off.sh  # ④ root で crontab の行をコメントアウト（退避つき）
```
- `host_cron_switch.sh` は**行ごとに** `cron_run.sh <cron_job_id> ` を含む行だけを対象にし（`KEIBA-VPN-ALL` のように多数のジョブが共有するタグはタグだけでは選ばない）、行頭に `#KEIBA-CUTOVER:<id># ` を付けるだけで元の行は変えません。対象が 0 行、または `--max-lines`（既定 4）超なら何もしません。`can-stop`（担当が渡っていて、ホストに残すジョブでないこと）を通らないと止めません（終了コード 5）。
- crontab の退避は `data/local/meta/cron_backups/crontab_before_cutover_<action>_<id>_<時刻>.txt`。`job_owner.json`／`fetch_owner.json` も `cron_backups/` に退避してから書き換えます。
- **ホストに残すと決めたもの**（バックフィル・モデリング・遠隔アクセス・stg のデプロイ・監視ポータルの死活など）は停止対象外で、`job_owner set` も `cutover_plan make` も拒否します。

### 4.6 停止後 1〜2 週間の観察と、ロールバック
- 観察: 毎日 `cron_cutover_status`（合格のまま）／Slack の結果通知・スクレイピング状況・日次コストが途切れない／`vps-watchdog` に異常が出ない。
- **ロールバック**（どの段階でも。順序は「行を戻す → 担当を外す」）:
  ```bash
  sudo bash /tmp/cutover_drift-monitor-weekly_rollback.sh
  # = host_cron_switch.sh enable（行を戻す）→ job_owner unset（担当を外す）。VPS の取得は fetch_owner.json の agent_tasks からも外す
  ```
  GCP 側を止めたいときは、そのグループの Scheduler を `gcloud scheduler jobs pause keiba-<グループ>`（削除しなくてよい）。取得系は `agent_tasks` から外せば実行役が拾わず、ホストの `auto_scrape.sh` が実行を再開します。
- 戻した直後は、GCP とホストが**二重に動く**可能性があるので、`pause` を先にするか、並走（影モード）のまま観察します。
- 全面的に戻す場合: `config/job_owner.json` の `owners` を `{}` にし、crontab を退避（`cron_backups/`）から `crontab <ファイル>` で復元。

### 4.7 開催日のレース別取得（`raceday-runner` の置き換え）の切替だけは時刻に注意
`reserve-race-day` が前夜 19:10／19:40 に翌日の全レースのタスクを積みます。**この 2 つの tick の間か、その後〜翌朝 07:30 までに**切り替えてください（積む前に担当を渡すと、翌朝のホストの runner が止まり、タスクは積まれず、取得が空く）。開催日のない日に切り替えるのが最も安全です。

---

### 4.8 影モード（VPS の取得をホストと並走で検証。取得ジョブの切替の前提）
取得系（`entries-sync`・`race-odds-*`・`race-weight-check`・`daily-race-lists` など VPS の実行役に移すもの）は、GCP のジョブと違って**ホストと同じ netkeiba を叩く**ので、parity（成果物の一致）の代わりに「並走で数日間見届ける」を基準にする。
- 仕組みと手順: `vps_agent.md` 5.5 章（有効化 `KEIBA_FETCH_SHADOW_ENABLED=1`、対象は `config/fetch_shadow.json`、書き込みは `shadow/<日付>/` だけ、1 日の上限・ジッター・アクセス制限での即停止、日次集計 `fetch-shadow-report`＝毎朝 07:30 JST）。
- 切替の関門: `cron_cutover_status` の `shadow_days` → `cutover_plan make`（`--force` でも作らない）→ `_1_owner.sh` の先頭（`fetch_shadow can-switch`）→ `job_owner set` → `host_cron_switch.sh disable`（`can-stop`）。**どこを通っても、基準を満たす前は止まる**。
- DEC-023: 影モードの間は netkeiba への取得がほぼ 2 倍になる（平日 約 250 件/日、開催日 約 1,000 件/日の見積り）。ジッターと上限で重ならないようにし、ホストのバックフィルと同じ時間帯に走らせない。
- ロールバック: `KEIBA_FETCH_SHADOW_ENABLED=0`（ホストの取得は止めていないので他は不要）。

## 5. 運用（切替後）
| 項目 | 内容 |
|---|---|
| 記録 | `ops/job_runs/<ジョブ>/…`（開始・終了・遅れ・結果・ピークメモリ・宣言メモリ）。メモリは並走期間の実測で `memory_mib` を見直す |
| 通知 | 失敗＝緊急、allow_fail の一部失敗＝注意（`alerting.notify`→`send_payload`）。VPS・取得は `vps-watchdog`。コストは `gcp-daily-cost-report`（想定 vs 実績・急増警告） |
| 費用 | Scheduler 13 本（約 ¥150）＋ Run Jobs（無料枠内の見込み。`event-pump` は開催日の日中のみ毎分）＋ Cloud Tasks（無料枠）。`architecture_target.md` 4 章の計算を `cost_estimate` で更新 |
| ジョブの追加 | `config/gcp_job_map.json` に足す → `validate` → `scheduler_gen`。新しい Python は `-m` 形式のみ（許可リスト）。メモリは宣言値で守る |
| モデルの更新 | ホストで学習 → `model_publish publish`。ジョブは次回の起動で検証つきに取得（イメージの作り直し不要）。`behavior_flags` の変更は `model_release_checklist.md` |

---

## 6. 未解決・リスク（正直に）
1. **Slack 通知系（`slack-*`）の元データ**: 現行はホストのローカル（`data/local/meta/scrape_*`、`prediction_results`）を読む。GCP で動かすには、取得状況は VPS 実行役の記録（`ops/fetch_tasks/`）、予測成績は `state/` から読む形への小改修が必要。それまで `ready=false`（ホストの通知が続く）。
2. **VPS 実行役の `state_out`**: `race_lists`・`race_day_schedule`・`cushion` はローカルのみのデータ。実行役がタスク完了時に `state/` へ上げないと `reserve-race-day` が動かない（vps-agent 側の作業）。
3. **`raceday-result-runner`**（T+15 の速報結果の常駐ループ）は VPS の許可リストに無く、ホストに残している。個別タスク化したら `vps` に移す。
4. **state／メモリの実測は見積り**: `state_in` の上限と `memory_mib` は見積りで、並走期間に Cloud Run の実測と `state_sync plan` で置き換える。`feature-accumulate`（feature_store 約 1GB 級）と `drift-monitor-weekly`（full エンジン）は 4Gi に収まらない可能性がある。
5. **`final_odds` と各モデルの `local_path`** はコードの既定から推定した値。ホストで `model_publish status` を見て直す。
6. **Scheduler のロールの前提**: `run.invoker` に `run.jobs.run` が含まれるかは未確認（`gcp_setup_runbook.md` 3 章で確認）。含まれなければ Job ごとに `run.developer`。
7. **`cron_run.sh` の変更はホストの全 cron 行に影響**する。担当が空の間は従来どおり実行する作りだが、反映時に 1 日様子を見る。
8. **`horse-remarks-daily` は 03:00 JST（メンテナンス窓と重なる）**: 移行とは別に時刻の見直しを検討。
9. 予測成績・ドリフト・凍結追跡は、2026 年を**最終確認のみ**に使う方針（テストセット②）と整合する。移行でデータの選択・学習は変わらない。
