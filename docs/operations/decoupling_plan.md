# ホストPC(stg)からの完全な切り離し: 依存の監査と置き換え計画

**作成**: 2026-10-22 / **種別**: 監査と設計（コードの機械検査を除き、ホスト・GCP・VPS・Cloudflare は変更していない）
**方針（ユーザー確定 2026-10-22）**: 公開(prod)は**ホストPC(stg)と何のつながりもない状態**にする。アプリの運用・データ取得・モデルの読み込み・監視は **VPS / GCP だけで完結**し、ホストPCは**モデルの再学習など研究専用**（運用に影響しない）。
**ただし順序の条件（確定済み）**: 「prod で数日間データ取得が正常に通るのを見届けるまで、ホストの取得は続ける」。切り離しはこの条件を満たしてから進める。
**関連**: [architecture_target.md](./architecture_target.md)（全体設計）・[migration_runbook.md](./migration_runbook.md)・[gcp_jobs_migration.md](./gcp_jobs_migration.md)・[vps_agent.md](./vps_agent.md)・[cloud_run_api.md](./cloud_run_api.md)・[env_parity.md](./env_parity.md)

---

## 0. 結論

1. **今の公開(prod)は、ホストに依存している**。画面(VPS)は動いても、**API・予測・データ・監視・配備がすべてホスト経由**で、ホストが止まると予測の表示も更新も止まる（`api-internal` がホストのトンネルを指すため）。
2. 依存は **20 項目**（高 10・中 7・低 3）。準備(コード・スクリプト・手順書)は大半が**すでに用意済み**で、足りないのは「実行して切り替える作業」と、**次の 4 つの設計判断**:
   - (a) 予測の置き場: **ホストのローカル出力 → GCS 正本**（serving_state の“ホストから push”を反転し、Jobs が GCS に書く）。
   - (b) ホスト→運用の受け渡しは **`model_publish`（モデルを GCS に置く一方向）だけ**にする。
   - (c) **配備のビルドをホストから外す**（CI/Cloud Build でビルド → GCS → VPS が pull）。
   - (d) **監視ポータルは Cloud Run へ**（VPS には置かない。メモリ予算 600MB を守るため）。
3. `gcp_job_map.json` の host 21 件＋retire 2 件を見直した結果、**研究専用は 5 件だけ**。**運用に必要なのに host になっていたのが 12 件**（`raceday-result-runner`・`horse-remarks-daily`・`backfill-2022〜2026`・`backfill-all-horse`・`horse-training-backfill`）で、VPS/GCP へ移す。廃止・置き換えが 6 件（`config/decoupling_roles.json`）。
4. **stg はホストから GCP 側のステージング**（Cloud Run のタグ付きリビジョン＋VPS の prod-test）に置き換える。
5. 完了の定義は **「ホストの API・トンネル・cron をすべて止めても、公開の全機能が続く」** を 6 章の試験で確認できること。機械検査は `python -m src.scripts.ops.host_dependency_scan`（8 章）。
6. 月額への影響は **数十〜200 円程度の増加見込み**（単価の一部は未確認、7 章）。第一案 A の上限（約 2,160 円）にすでに近いので、予算 2,000 円以内に収めるには保存量の削減（7 章）が要る。

---

## 1. 目標像と原則

```
研究（ホストPC）                         運用（公開 prod）
  学習・評価・ノートブック・MLflow         app.megu-keiba.com ── Cloudflare Access ── VPS(Next.js, nginx, cloudflared)
        │                                                                              │ 取得の実行役(pull)
        │ ① publish（一方向）                                                         ▼
        ▼                                                     GCS(正本) ◀──── Cloud Run API / Jobs(予測・集計・通知) ◀── Scheduler/Tasks
  models/<ver>/ + latest.json(sha256)                                │
        ▲                                                            ▼
        │ ② 読み取り専用の鍵(keiba-viewer)で必要なデータだけ取得     <MONITOR_HOST> (Cloud Run, Access)
```

原則:
1. **運用側からホストを呼ぶ経路をゼロにする**（ホスト→運用の“押し出し”だけ。①publish）。
2. **正本は GCS**。ホストのローカルファイルは正本にならない。運用のデータはすべて VPS/GCP が書く。
3. **ホストは運用データを書かない**。研究が読むのは読み取り専用の鍵（作成済みの `keiba-viewer`）。書くのは `models/` など publish 専用の接頭辞だけ（9.4 の権限分離）。
4. **環境名は prod 一つ**。確認用の stg は GCP 側（4.7）。
5. **ホストが止まっても公開が続くこと**を、試験(6 章)と機械検査(8 章)で保証する。

---

## 2. 実行時依存の一覧（監査結果）

重大度: **高** = ホストが止まると公開が止まる／データが古くなる、**中** = 一部機能が止まる／手作業が残る、**低** = 運用の道具。
場所は「ファイル:行」ではなく所在（行は変わるため）。機械検査の件数は 8 章（運用経路 108 ファイル・310 件。うち高重大度の直書き 26 件）。

### 2.1 高（10）

| ID | 依存 | 場所 | 理由 | 置き換え先 | 状態 |
|---|---|---|---|---|---|
| D-01 | **API がホスト**。VPS の nginx → `<API_INTERNAL_HOST>` → Cloudflare Tunnel(`<TUNNEL_STG>`) → ホストの FastAPI :23220 | `scripts/ops/vps/templates/nginx-keiba-ui.conf`・`render_nginx_access.sh`・`vars.env`、Cloudflare の Public Hostname | ホスト停止＝API 全停止 | **Cloud Run**（`switch_api_upstream.sh cloudrun`） | 準備済み・未実施 |
| D-02 | **予測の生成がホストの cron/常駐**（前夜保存 `raceday-eve`、馬体重検知後 `raceday-weight-runner`、T-60/T-15 の `raceday-runner`） | `src/pipeline/inference/weight_triggered_predict.py`（`save_cached`）、`scripts/cron/*`、ホストの root crontab | 予測が出なくなる | **GCP Jobs**（`predict-race`・`eve-precompute`）＋Cloud Tasks の予約。取得の合図は VPS 実行役 | 準備済み・blockers あり（モデル publish・serving_bundle） |
| D-03 | **API が読むデータの置き場がホストのローカル**（`data/local/inference/{predictions_v3,predictions_backfill,snapshots,lap_forecast,tracking_v2}`、`data/local/prediction_results`、`data/local/meta/entries_state`、`data/page_reference/**`）。Cloud Run へは“ホストから `serving_state push`”する手作業 | `src/api/local_predictions.py`・`prediction_results_api.py`・`entries_status.py`・`entry_card.py`・`config/serving_state.json` | ホストが push しない限り古い | **Jobs が GCS に書き、Cloud Run が読む**（`serving_state` の push 元をホスト→Jobs に反転。4.1） | 設計のみ |
| D-04 | **取得（出馬表・オッズ・馬体重・レース一覧・結果の取り直し）の cron がホスト**、秘密(netkeiba ログイン)がホストの `.env` | ホストの root crontab、`scripts/auto_scrape.sh` | データが入らない | **VPS の取得の実行役**（影モードで検証→担当を移す） | 準備済み・影モード実装済み・**ホスト継続中** |
| D-05 | **結果の速報取得 `raceday-result-runner`**（発走 T+15）がホスト。job_map では host 分類 | ホストの cron `30 22 * * *`、`scripts/cron/run_auto_scrape_logged.sh` | 結果→予測成績・結果通知・履歴特徴量が止まる。**運用に必要なのに host になっていた** | VPS 実行役のレース別タスク＋GCP Job `slack-results` | 未着手（要タスク定義） |
| D-06 | **モデル・バンドル・前段/補助・serving 状態がホストのパス** | `config/serving_state.json`（`KEIBA_V3_BUNDLE` 等の `local_path`）、`data/local/modeling/**` | モデルを読めない | **`model_publish`（GCS に置く）→ `latest.json`(sha256) を Jobs/Cloud Run が検証つきで取得**（4.2） | コード済み・初回 publish が未実施 |
| D-07 | **画面のビルド・昇格・配備がホスト**（`next build`、`promote.sh`、`maintenance_deploy`、`stage_deploy`） | `scripts/ops/vps/build_release.sh`・`promote.sh`、`src/scripts/ops/{stage_deploy,maintenance_deploy}.py`、ホストの root cron | ホストが止まると配備できない（公開は動くが更新できない） | **CI（Cloud Build 等）でビルド→GCS→VPS が pull**（`pull_release` 実装済み）。API は `gcloud run deploy`（4.8） | 部分的に準備済み |
| D-08 | **監視ポータル `<MONITOR_HOST>` がホスト :23900**（Tunnel `<TUNNEL_STG>` 経由）。ホストのファイル（`cron_runs`・`quality_health`・`page_quality`・`model_perf`）を直接読む | `src/monitor/app.py`・`cron_status.py`・`cron_failure_patterns.py`・`model_perf.py` | 監視が見えなくなる＝異常に気づけない | **Cloud Run**（最小 0）＋GCS から読む（4.6） | 未着手 |
| D-09 | **秘密情報がホストの `.env`**（netkeiba ログイン・GCS 鍵・Slack Webhook・`DEV_PASSWORD`/`DEV_SECRET_KEY`・Cloudflare トークン） | ホストの `.env`（root 所有 640） | ホストが運用の鍵の保管庫になっている | **Secret Manager**（GCP 側）＋VPS の `.env`（netkeiba・Cloudflare のみ）。ホストには研究に要るものだけ | 手順書あり・登録は未実施 |
| D-10 | **ログインのセッション署名・会員名簿**（`DEV_SECRET_KEY`、`config/members.json`）がホストの値とリポジトリ | `src/api/auth.py`・`flask_auth.py` | セッションの互換が切り替え時に切れる／会員名簿がイメージ内 | Secret Manager に**同じ値**を登録して切替。名簿は当面イメージ内（更新＝再デプロイ）、会員が増えたら GCS/Firestore へ | 設計済み（cloud_run_api.md） |

### 2.2 中（7）

| ID | 依存 | 場所 | 置き換え先 |
|---|---|---|---|
| D-11 | Slack の結果・状況・異常通知が**ホストの `data/local/meta/scrape_*`・`prediction_results`** を読む | `src/scripts/data/{daily_results_notify,scrape_status_notify}.py` | GCP Jobs（`slack-*`）。元データは VPS 実行役の結果記録(GCS)と Jobs の出力に置換（job_map の blockers） |
| D-12 | 集計がホストの cron: `jockey-trainer-stats`・`export-tables-daily`・`serving-state-precompute`・`feature-accumulate` | job_map（gcp・ready=false） | GCP Jobs（state の実サイズ確認→実測で 4Gi に収まるか） |
| D-13 | **PostgreSQL**（`sync_pg_from_gcs`・`ensure_postgres`、API の DB 経路 41 箇所） | `src/api/app.py`・`flask_app.py`・`v1/routes/analytics.py`・`src/monitor/*` | **廃止**（Cloud Run では DB 経路は 503）。必要な集計は Jobs が GCS に出す |
| D-14 | ラップ予想・追走難度の前夜保存がホスト | `lap_forecast_service.py`・`tracking_v2_service.py` | Jobs（`lap-forecast-*`・`eve-precompute`）＋モデル publish |
| D-15 | ドリフト監視・凍結追跡・データヘルス・ページ品質・オッズドリフトがホスト | `src/scripts/data/drift_*`、`src/data_health`、`src/monitor/page_quality_check.py` | Jobs（`drift-monitor-weekly`・`frozen-tracking-weekly`・`page-quality-check`・`odds-drift-weekly`〔ready〕） |
| D-16 | **バックフィル群**（`backfill-*`・`backfill-all-horse`・`horse-training-backfill`）が全部ホスト | ホストの cron | 3 章の層分け（当年・直近年は VPS/GCP、古い年は研究） |
| D-17 | 取得の日次タスクのうち VPS 未移行: `horse-remarks-daily`・`new-horse-pedigree`・`raceday-comment`・`raceday-evening`・`raceday-eve-fetch`・`weekly-update` | job_map（vps・ready=false）、`fetch_owner.json` | VPS 実行役（250M での実測後に段階移行） |

### 2.3 低（3）

| ID | 依存 | 場所 | 置き換え先 |
|---|---|---|---|
| D-18 | ホスト用の運用道具: `stage_deploy`・`maintenance_deploy`・`server_watchdog`・`tunnel_stg_ui.sh`・`fetch_stg_health.sh`・`enable_v5_shadow.sh`・ホストの Cloudflare トンネル | `src/scripts/ops/`・`scripts/server/` | 廃止（D-07 の置き換えに含む）。ベースラインから 1 つずつ消す |
| D-19 | **stg 環境**（`KEIBA_ENV=stg`、`<STG_HOST>`、ホストのトンネル、`prod-test`） | `src/config/deployment.py`、Cloudflare | GCP 側のステージング（4.7） |
| D-20 | 研究コードがホストの絶対パス（`<HOST_HOME>/work/tp*_out`、`notebook_utils.py`） | `src/scripts/data/tp*`、`src/pipeline/models/notebook_utils.py` | **対象外**（ホストは研究専用なので問題なし）。運用経路から参照しないことだけ機械検査する |

> `data/local` の参照の多く（機械検査で 174 件）は、`HybridStorage` のローカルキャッシュ層（GCS が正本、ローカルは読み書きの高速化）で、**実行環境に空のディレクトリがあれば動く**ので“ホスト依存”ではない。真に問題なのは、**GCS に無く、ホストにしか正本が無いデータ**で、それが D-03/D-06 に挙げたもの。

---

## 3. 「ホストに残す」ジョブの見直し（`config/decoupling_roles.json`）

`gcp_job_map.json` の host 21 件＋retire 2 件を、最終形の観点で分類し直した（テスト `tests/ops/test_decoupling_roles.py` が job_map との食い違いを検出する）。

| 分類 | 件数 | ジョブ |
|---|---:|---|
| **研究専用（ホストに残す）** | 5 | `modeling`・`sporthorse-births-nightly`・`backfill-2020-fast`・`backfill-2021-fast`・`keiba-remote-access` |
| **運用に必要 → VPS へ** | 10 | `raceday-result-runner`・`horse-remarks-daily`・`backfill-2022〜2026-fast`・`backfill-2024-full`・`backfill-all-horse`・`horse-training-backfill`（当年分） |
| **運用に必要 → GCP へ** | 2 | `backfill-2025-full`・`backfill-2026-full`（full は重い。取得の GCP 試験の合格後。不合格なら分割して VPS） |
| **廃止・置き換え** | 6 | `maintenance-deploy`・`server-watchdog`・`rotate-logs`・`ensure-monitor`・`sync-pg-from-gcs`・`ensure-postgres` |

判断の根拠:
- **バックフィル**は「過去データの取りこぼし補填」。公開の予測は**出走馬の履歴**を読むので、直近数年の欠けは精度と表示に直結する（運用に必要）。古い年(2020〜21)は学習データの完全性のためで研究。
- **training（調教）**は、当年分が公開の特徴量（調教被覆）に直結する。過去年の取り直しは研究。
- **sporthorse 誕生日**は血統研究用（外部サイトのセッション依存）で、公開経路は読まない。必要になったら成果物を GCS 経由にする。
- ホスト→GCS の**書き込みを研究側に残す場合は**、運用の正本とは別の接頭辞(`research/`)に分け、運用は読まない（9.4）。

---

## 4. 置き換え設計

### 4.1 データ: GCS 正本化と、serving_state の反転
- 現状: Cloud Run へは“ホストが `serving_state push`”（手作業）。**これが最大の隠れた依存**。
- 変更: `serving_state` の生成元を **Jobs（`serving-state-precompute`・`export-tables-daily`・`predict-race` の出力）**にする。Jobs は出力を GCS の `serving_state/` と正本の接頭辞（`race_predictions` など）に書き、Cloud Run は起動時に pull（実装済み）。
- 取得で作る `data/page_reference/**`（race_lists・race_day_schedule・cushion など）は、**VPS 実行役が GCS に直接書く**（`state/` ではなく正本の接頭辞）。Jobs はそれを読む。
- 完了条件: `serving_state push` をホストで実行していない状態で、1 週間、Cloud Run の表示が更新され続ける。

#### 4.1.1 置き場所の決定（2026-10-24、ユーザー依頼: VPS か GCP のどちらかに置く。コストとレイテンシを見て決める）
**決定: 正本は GCS（GCP）。VPS には置かない。** VPS は画面と薄い取得役だけで、API 用のデータは持たない。
- 理由（コスト）: 日次の予測成績は 1 日 約 150KB・年 約 15MB、予測 parquet は 1 日 約 150KB。保存・読み取りとも GCS の最小課金（月 数円以下）。VPS に置くと API を VPS で動かすことになり、常駐 600MB の上限（architecture_target.md）と衝突する。
- 理由（レイテンシ）: Cloud Run は永続ディスクを持たないため、ローカル読みは GCS 読みでしか置き換えられない。GCS の 1 オブジェクト読みは同一リージョンで数十 ms。**確定済みの日は変わらないのでプロセス内に 6 時間キャッシュ**、当日分は 60 秒。ローカルがあれば（ホスト・テスト）従来どおりローカルが先で、GCS は“無いときだけ”。さらに VPS の nginx で GET をキャッシュする案は 5 章（別途）。
- 置き場: `gs://<GCS_BUCKET>/ops/prediction_results/<YYYYMMDD>.json`（運用の正本の接頭辞。`research/` とは分ける）。
- 実装: `src/pipeline/inference/prediction_results.py`（`gcs_enabled` / `read_day` / `write_day` / `list_days` / `sync_to_gcs`）。**有効条件は `KEIBA_PREDICTION_RESULTS_GCS=1`、または Cloud Run/Jobs（`KEIBA_RUN_LOCATION=gcp`）。0 で無効、dev は常に無効**。書き込みの失敗はローカル保存を止めない。
- 初回の移行: ホストで `.env` に `KEIBA_PREDICTION_RESULTS_GCS=1` を足し、`python -m src.scripts.data.prediction_results --sync-gcs`（86 日・約 15MB を 1 回）。以後は集計のたびに両方へ書く。Cloud Run には serving_state の `data/local/prediction_results/**` は不要になる。
- 未完了（同じ方針で順に移す）: `entries_state`（出馬表の段階）、`events`（予測の経過）、予測 parquet（`predictions_v3`）。予測は `KEIBA_V3_SERVING=predict` で `race_predictions`（GCS）に保存される経路が既にあり、shadow→predict の切替（モデルの採用判断）が前提。

### 4.2 モデル: ホスト→運用の一方向の契約
研究（ホスト）が運用に渡すのは**これだけ**。
```
gs://<bucket>/models/<component>/<version>/        # バンドル本体（manifest.json を含む）
gs://<bucket>/models/<component>/latest.json       # {version, sha256, created_at, feature_manifest_sha, train_end, notes}
```
- `python -m src.ops.model_publish publish --component <名前>`（実装済み）。**latest.json は最後に書く**（書いた瞬間が昇格）。
- 運用側（Jobs/Cloud Run）は**起動時に latest.json を読み、sha256 を検証して取得**する（実装済み）。検証に失敗したら**直前の版を使い続け**、Slack に注意を出す。
- **特徴量の列順・名前は `manifest.json` で検証**（学習側と推論側で同じビルダー。env_parity の `release_manifest` に版を含める）。
- 運用がホストへ問い合わせる経路は**作らない**。ホストが止まっていても、既存の `latest.json` で動き続ける。
- 版の切り替え（v5→v6 など）は、publish → stg(GCP ステージング)で確認 → `latest.json` の“採用版”ポインタを進める、の 2 段（`model_release_checklist.md`）。

### 4.3 取得
- **VPS の実行役**（60 秒 pull、`MemoryMax=250M`）が担当。タスク表は GCS の `ops/fetch_tasks/`。ホストの cron は影モード(`shadow/`)で並走検証→担当を移す（`fetch_owner.json`。`cron_cutover_status` の関門を満たすまで停止手順は作られない）。
- 秘密(netkeiba ログイン)は VPS の `.env` のみ。GCP には置かない。
- 取得が GCP から通るかの到達試験（`gcp_smoke_results.md`、3 回）は**取得を GCP に寄せるかどうか**の判断材料で、**切り離しの必須条件ではない**（VPS だけで完結できる）。ただし `backfill-full` のような重い取得は GCP Jobs に寄せたい。試験が不合格なら**分割して VPS で低頻度に回す**。
- VPS の IP が遮断された場合の代替は、これまで“ホスト(自宅 IP)に戻す”だったが、**最終形ではホストに戻せない**。代替案: (a) 取得間隔を延ばす、(b) 別の VPS を従量課金で一時的に立てる（月数百円）、(c) 自宅回線のホストを“研究用の取得専用機”として**運用とは別の接頭辞に書き、必要なときだけ手動で運用に取り込む**（切り離しの原則を弱める）。9 章で決める。

### 4.4 予測生成・集計・通知
- 予測: Cloud Tasks が前夜に予約した時刻に Jobs `predict-race` を起動（T-60/T-15、馬体重検知後の即時予測は `event-pump`）。出力は GCS の `race_predictions`。
- 集計・通知・ドリフト・凍結追跡・データヘルス: Scheduler＋Jobs（job_map の gcp 群）。**通知は【PROD】（`KEIBA_RUN_LOCATION=gcp`）**。
- 直前オッズ(T-10/T-5/T-2)は VPS 実行役が取り、GCS の `race_odds.snapshots` に積む（既存設計）。

### 4.5 API・UI・認証
- API: Cloud Run（最小 0・最大 3・1 ワーカー・2GiB）。共有ヘッダ `X-Keiba-Proxy-Key` と Access のサービストークン（準備済み）。
- UI: VPS の Next.js。上流は `switch_api_upstream.sh cloudrun`。**`api-internal`（ホスト向け）の Access アプリ・Public Hostname・Tunnel `<TUNNEL_STG>` のルートは、切替後に削除**する。
- 認証: `DEV_SECRET_KEY`/`DEV_PASSWORD` を Secret Manager に**ホストと同じ値**で登録（切替でセッションを切らない）。会員名簿 `config/members.json` は当面イメージ内。`KEIBA_TEST_USER` は公開前に off（`tiered_access.md`）。

### 4.6 監視ポータルの置き場（Cloud Run か VPS か）

| 観点 | Cloud Run（最小 0） | VPS に同居 |
|---|---|---|
| メモリ | VPS を使わない（VPS 予算 600MB を守れる） | Flask＋表の描画で +80〜150MB。通常時 500〜580MB → 約 660〜730MB で**目標超過**（`MemAvailable` 1GB 維持も厳しくなる） |
| 可用性 | VPS 障害時も見える（**VPS の死活を見るポータルが VPS と一緒に落ちない**） | VPS が落ちると監視も落ちる |
| 費用 | 無料枠内の見込み（リクエスト少）。コールドスタート 5〜15 秒 | 0 円 |
| 実装 | GCS から読むように改修が必要（現状はホストのローカルファイル。`cron_runs` 等を Jobs/VPS が GCS に出す） | 改修は同じ（GCS を読む）。VPS に置くメリットは薄い |
| 認証 | Cloudflare Access の背後（現状と同じ。IAP 不要） | 同じ |

**推奨: Cloud Run**。理由は上の 2 点（VPS のメモリ予算を守る・VPS の死活を VPS の外から見る）。改修の中身は「ホストのローカルファイルを読む箇所を、GCS から読む（または Jobs が出す JSON を読む）」ことで、`model_perf` は既に“事前計算キャッシュを読むだけ”の作り（キャッシュの置き場を GCS にするだけ）。移すまでの間、ホストの monitor は“研究者向け”として残してよいが、**運用の監視には使わない**。

### 4.7 stg の置き換え
- 目的: 本番に出す前の確認（画面・API・モデルの切替）。ホストの stg は使わない。
- **API**: Cloud Run の**タグ付きリビジョン**（トラフィック 0%、`https://r<TAG>---…`）。`cloudrun_smoke`＋`env_parity_check` で確認してから `promote`。
- **画面**: CI でビルドした成果物を VPS の `<PROD_TEST_HOST>`（Access 保護）で**切替前に起動して確認**する（メモリは一時的に +100MB。確認が終わったら止める）。確認後に本番へ `install_release.sh`。
- **モデル**: `latest.json` の“採用版”を進める前に、Jobs の手動実行（ステージング用の出力接頭辞 `staging/`）で並走し `parity_check` で比較。
- env_parity: stg と prod の比較は「**ステージング(新リビジョン) と 現行 prod**」の比較に変わる。`release_manifest` はそのまま使える（マニフェストの取得先が Cloud Run のタグ URL になる）。
- `KEIBA_ENV=stg` の環境は**廃止**し、`prod` の一つにする（9 章の判断）。

### 4.8 配備（ホストを使わない）
- UI: CI（Cloud Build の無料枠 2,500 分/月、または GitHub Actions）で `build_release.sh`（“ホスト機か CI で実行”できる作りになっている）→ tarball＋sha256 を GCS の `releases/ui/` に置き、`latest.json` を書く → VPS の `keiba-release-pull-ui.timer`（毎日 03:10 JST）が取得・検証・`install_release.sh`・`healthcheck.sh`・失敗時ロールバック（実装済み）。
- API/Jobs: `gcloud run deploy`（`deploy_cloud_run_api.sh`、ユーザーが Cloud Shell で、または CI から `keiba-deployer` で）。
- 窓内自動デプロイ(`maintenance_deploy`)は廃止。VPS の pull 型が窓(03:00〜04:00)に動く。**窓の中は “/ ” が 503 のメンテナンス画面になるので、確認 URL は窓の対象外（`/login`）にする**（2026-10-22 の失敗の教訓。`server_watchdog` を修正済み）。

---

## 5. 移行順序（フェーズ D0〜D6）

“ユーザー”＝あなたの作業、“Claude”＝私が用意・実行する作業。**各フェーズの完了条件を満たすまで次へ進まない**。ロールバックは数分〜数十分で戻せるものだけを順に。

| フェーズ | 内容 | ユーザー | Claude | 完了条件 | ロールバック |
|---|---|---|---|---|---|
| **D0** 準備（ほぼ済） | UI を VPS・Cloud Run/Jobs/実行役/影モード/pull デプロイのコード・手順書・GCP の土台（API・SA・予算・課金エクスポート） | 到達試験 3 回、Secret Manager の登録、Cloudflare Access の整理 | コード・テスト・手順書 | 到達試験の記録、秘密の登録完了 | — |
| **D1** API を Cloud Run へ | `api-internal` の上流をホスト→Cloud Run。**serving_state はまだホストから push（暫定）** | `setup_cloud_run_api.sh`→`serving_state push`→`deploy --apply`→`promote`→VPS で `switch_api_upstream.sh cloudrun` | `cloudrun_smoke`（ホストとの出力一致）、スクリプト修正 | 1 週間、表示がホストと一致し安定 | `switch_api_upstream.sh host`（数秒） |
| **D2** モデル publish と予測生成 | 初回 `model_publish`、Jobs `predict-race`・`eve-precompute`・`lap-forecast-*` を並走→切替。**serving_state の生成元を Jobs に反転（4.1）** | モデルの publish、Scheduler/Queue の作成、`job_map ready` | blockers の解消、`parity_check` の運用 | 同じ日をホストと GCP で処理して一致、`cron_cutover_status` 合格 | ホストの cron を戻す（`_2_cron_off` の対を実行） |
| **D3** 取得の移行 | 影モード 5 日（開催日を含む）合格→担当を VPS に。`raceday-result-runner`・`horse-remarks-daily`・`backfill-*`(運用分) も VPS 実行役へ。**ホストの取得は“prod で数日正常”を見届けるまで継続** | 影モード有効化、`fetch_owner` の移行、IAM 条件 | タスク定義の追加(D-05/D-16/D-17)、実測、`shadow_eval` | 直近 5 日合格・開催日の T-60/T-15 が全レースで成功・Slack 異常ゼロ | 担当をホストへ戻す |
| **D4** 通知・集計・監視を GCP へ、monitor を Cloud Run へ | Slack/集計/ドリフト/データヘルスの Jobs 切替、監視ポータルを Cloud Run 化（4.6） | Cloud Run のデプロイ・Access の付け替え | monitor の GCS 読みへの改修、Jobs の blockers 解消 | 1 週間、通知が【PROD】で届き、monitor が GCP 上で見える | ホストの monitor・cron に戻す |
| **D5** 配備の脱ホスト・stg 置換 | CI でビルド→VPS が pull、API はタグ付きリビジョン、`prod-test` で画面確認。`api-internal`・`<TUNNEL_STG>`・ホストの Access アプリを削除 | CI の用意（Cloud Build トリガ）、Cloudflare の整理 | CI 設定、`promote` の CI 化、`maintenance_deploy` 等の廃止 | ホストを使わず 1 回、UI と API を更新できた | 旧手順（ホストでビルド）に戻す |
| **D6** ホストを研究専用に | ホストの運用 cron をすべて停止（研究専用の cron のみ残す）、**鍵の権限分離(9.4)**、ホスト停止試験(6 章)を実施 | cron 停止の実行(root)、鍵の付け替え、試験の実施・確認 | `cutover_plan`、`host_dependency_scan` のベースラインを 0 に近づける | 6 章の試験に合格、1〜2 週間の観察で異常ゼロ | 停止した cron を戻す（再有効化スクリプト） |

順序の注意:
- **D1 は D2 より先でよい**（API を先に Cloud Run へ移しても、データは暫定でホストから push）。ただし**D1 のままホストを止めてはいけない**（push が止まる）。
- **D3 の「ホストの取得を止める」は最後**。ユーザーの条件（prod で数日間正常を見届ける）を満たすまで続ける。
- D2・D3 は並行でよいが、**同じタスクを両方が取る時間を作らない**（`fetch_owner` の担当で排他）。

---

## 6. 「ホストを止めても公開が続く」ことの検証（ホスト停止の模擬試験）

実施: D6 の前と後に 1 回ずつ。**前**は“止めたつもり”で（ホストの cron/トンネルを止めて）確認し、**後**は本当にホストを止める（ホストを停止または別ネットワークへ）。

**準備（ホスト側。ユーザーが実行）**
1. ホストの root crontab を退避して全行をコメントアウト（`host_cron_switch.sh disable all` 相当。研究専用の行は別に残す）。
2. ホストの FastAPI/Flask/Next.js を停止（`server_watchdog` も止める）。ホストの Cloudflare トンネル(`<TUNNEL_STG>`)を止める。
3. ホストの `.env` から GCS 鍵・Slack・netkeiba を退避して一時的に使えなくする（任意。秘密のホスト依存を洗い出す）。

**確認（Claude が可能なものは代行。ホストの読み取りは不要）**

| # | 機能 | 確認方法 | 合格 |
|---|---|---|---|
| 1 | 画面の表示 | `https://app.megu-keiba.com/` を匿名・会員で開く | 200。匿名は勝率/複勝率のみ、会員は期待値・めぐAI印・推奨馬券が見える |
| 2 | 予測の表示 | 直近の開催日のレース詳細 | 予測・期待値・印・展開が出る（欠けたら“未計算”でなく表示） |
| 3 | 推奨馬券 | 会員で `POST /api/v1/betting/optimize` | 候補が返る（404/503 でない） |
| 4 | 出馬表の更新 | 木曜の `entries-sync` の時間帯 | 出馬表の `phase` が更新される（VPS 実行役→GCS→Cloud Run） |
| 5 | オッズの更新 | 開催日の T-60/T-15 | `race_odds` のスナップショットが積まれ、画面のオッズが変わる |
| 6 | 馬体重→予測 | 馬体重の発表後 | 予測が自動で出る（Jobs が起動） |
| 7 | 結果の取り込み | 発走 T+15 | 速報結果・予測成績が更新される（D-05） |
| 8 | 通知 | 夕方の結果通知・取得状況・異常の模擬 | **【PROD】**が Slack に届く |
| 9 | 監視 | `https://<MONITOR_HOST>/models` ほか | 見える（GCP 上） |
| 10 | 死活 | VPS のハートビートを 15 分止める | `vps-watchdog` が緊急通知を出す |
| 11 | モデル更新 | ホスト停止中に `latest.json` が無くても動く | 既存の版で動き続ける |
| 12 | 配備 | ホストを使わず UI を 1 回更新 | CI→GCS→VPS pull が成功 |
| 13 | 機械検査 | `python -m src.scripts.ops.host_dependency_scan --strict --summary` | 運用経路の `host_*`・`api_internal`・`postgres` が 0（`local_data` はキャッシュ層のみ） |
| 14 | 逆の確認 | ホストに運用データが書かれていないこと | ホストから GCS の運用接頭辞に書けない（権限分離 9.4） |

---

## 7. 費用への影響（月 2,000 円以内）

第一案 A は **約 300〜2,160 円**（`config/cost_assumptions.json`）で、上限はすでに予算に近い。切り離しで増える項目（見込み。単価の一部は未確認）:

| 項目 | 増分（月） | 根拠・備考 |
|---|---:|---|
| 監視ポータルの Cloud Run（最小 0） | 0〜50 円 | 無料枠内の見込み |
| 取得を GCP/VPS に寄せる分の Jobs（backfill 等） | 0〜200 円 | 週次・低頻度なら小さい。full が 4Gi・数十分なら上振れ |
| CI（Cloud Build） | 0 円 | 無料枠 2,500 分/月 |
| GCS の保存量（運用データの正本化・ホストが持っていた分が GCS に増える） | 50〜300 円 | 100GB あたり約 345 円。**増える量は未確認**（ホストの `data/local` の運用分のサイズを `state_sync plan` で測る） |
| ホストが GCS から読む（研究用の取得） | 注意 | GCS→外部の送信は約 18 円/GB。**全件ダウンロードは避け、差分だけ**（読み取り専用鍵で必要分のみ） |
| 取得 IP の代替（VPS 遮断時の保険） | 0〜数百円 | 必要になったときだけ |

**結論**: 合計の上限が約 2,160 円を超えうる。2,000 円に収めるには、(1) 古い生データの Nearline 化/圧縮（`architecture_target.md` 9 章 7）、(2) Scheduler のジョブ数を 6〜8 に絞る、(3) backfill を低頻度にする、のどれかが要る。実績は `gcp_daily_cost_report --watch` の“想定 vs 実績”で毎日見る。

---

## 8. 機械検査: `host_dependency_scan`

`python -m src.scripts.ops.host_dependency_scan [--strict] [--summary] [--update-baseline]`（`src/scripts/ops/host_dependency_scan.py`、ベースライン `config/host_dependency_baseline.json`、テスト `tests/scripts/test_host_dependency_scan.py`）。

- **対象**: 運用経路のコード（`src/api`・`src/utils`・`src/ops`・`src/monitor`・`src/scraper`・`src/pipeline/inference`・`src/scripts/{ops,cloud_jobs}`・`scripts/{ops,gcp,server}`・`frontend` の実装）。研究コード(`tp*`・`research`・notebooks・モデル学習)とテストは対象外。
- **検出する区分**: `host_path`(<HOST_HOME>)・`host_alias`(keiba-remote)・`host_port`/`host_port_literal`(stg の 232xx)・`api_internal`・`local_data`(data/local)・`page_reference`・`postgres`。
- **方式**: 現状（310 件・108 ファイル）をベースラインにして、**増えたときだけ失敗**する。依存を減らしたら `--update-baseline` で下げる。最終形（D6）は **高重大度（host_path / host_alias / host_port / api_internal）と postgres が 0**、`local_data`/`page_reference` が“キャッシュ層のみ”。
- 現状の高重大度（26 件）の多くは、**廃止対象の道具**（`stage_deploy`・`maintenance_deploy`・`tunnel_stg_ui.sh`・`fetch_stg_health.sh`・`enable_v5_shadow.sh`）に集中している（D-18）。これらを消せばほぼ 0 になる。
- `tests/scripts/test_host_dependency_scan.py::test_repo_has_no_new_host_dependencies` が、運用経路に**新しいホスト依存を足していない**ことを毎回の CI で確かめる。

---

## 9. 決めていただくこと（推奨値つき）

| # | 項目 | 推奨 | 理由 |
|---|---|---|---|
| 1 | 監視ポータルの置き場 | **Cloud Run**（4.6） | VPS のメモリ予算を守る・VPS の死活を外から見る |
| 2 | stg の扱い | **ホストの stg を廃止**し、GCP のタグ付きリビジョン＋`prod-test` に置換。環境名は `prod` 一つ | 研究専用化と整合。env_parity は“新リビジョン vs 現行”に変わる |
| 3 | VPS の IP が遮断されたときの代替 | まず取得間隔を延ばす。恒久策は**従量課金の予備 VPS を必要時だけ** | ホストに戻すと切り離しの原則を崩す |
| 4 | ホストの GCS 権限 | **読み取り専用(`keiba-viewer`)＋`models/` への publish 専用の書き込み鍵**（別 SA）。運用の正本には書かせない | 研究の誤操作が公開に影響しない |
| 5 | ホストが書く研究用データの置き場 | **別バケット(`…-research`)**。運用バケットとは分ける | 4 と同じ。ライフサイクルも分けられる |
| 6 | `backfill-*`（運用分）の移し先 | **VPS 実行役の低頻度タスク**。full は GCP 試験の合格後に Jobs | メモリ 250M に収める |
| 7 | 会員名簿の置き場 | 当面イメージ内。会員が増える段階で GCS/Firestore | 今は数人 |
| 8 | D6 の時期 | 影モード合格＋Cloud Run 安定 1 週間の後 | ユーザー確定の条件 |
| 9 | 古い生データの保存方針 | 2020〜2022 は Nearline/圧縮（研究が読むときだけ） | 費用（7 章） |

### 9.4 権限分離の案
- `keiba-viewer`（作成済み・読み取り専用）: 研究が運用データを読む。ホストの `~/.config/keiba-gcp/viewer.json`。
- `keiba-publisher`（新規）: `gs://<bucket>/models/**` への**書き込みのみ**（IAM 条件で接頭辞を限定）。ホストの publish 専用。運用の正本接頭辞（`race_predictions` 等）には書けない。
- ホストの `.env` から、運用の GCS 鍵（現状は書き込み可）・Slack・netkeiba・Cloudflare を**外す**（D6）。
- 運用側の SA（`keiba-api`・`keiba-jobs`・VPS）はホストの鍵と無関係。

---

## 10. 未解決・リスク（正直に）

- **取得の GCP 試験は 1 回目が netkeiba 側の 503 で無効**。結果次第で backfill の移し先（VPS/GCP）が変わる（3 章）。
- **VPS の 250M で収まる取得タスクの範囲は未測定**（`weekly-update`・`raceday-evening` など）。収まらないものは Jobs に寄せる必要があり、その場合は取得 IP の問題に戻る。
- **`serving_state` の反転（4.1）は設計のみ**。Jobs が出す全ファイルが GCS に揃うか、Cloud Run の起動時間（取り込み）が実測で許容内かは未確認。
- **予測の GCS 正本化**で、ホストの shadow/バックフィルの出力（`predictions_v3`・`predictions_backfill`）を GCS に置く扱いが要る（バックフィル分は一度 push すれば固定。新規は Jobs が書く）。
- **クラウドの単価**は一部未確認。保存量の見込みは `state_sync plan` で測るまで不明。
- 監視ポータルの GCS 読み込みへの改修は規模が読めていない（`monitor/app.py` は参照箇所が多い）。段階的に（まず `/models`・`/services` の主要ページ）。
- 研究側からの運用データの読み出しが GCS の送信料金（約 18 円/GB）を生む。**差分取得のルール**を決める必要がある。
