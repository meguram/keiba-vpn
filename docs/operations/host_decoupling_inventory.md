# ホスト切り離しの棚卸し: 定期実行の As-Is / To-Be（取得・予測・集計・通知）

作成: 2026-10-28。ホスト PC（stg）の root crontab と `config/gcp_job_map.json` を突き合わせた全量（#1〜#50）。
**To-Be は提案であり未決定。** 決まった内容は `config/gcp_job_map.json` と `config/fetch_shadow.json` に反映する。
関連: `docs/operations/architecture_target.md`（方針）、`docs/operations/vps_agent.md`（取得の実行役・影モード）、`docs/operations/cost_recalc_20261028.md`（費用）。

- As-Is の時刻は JST。ホストの cron は UTC なので +9 時間して直した。
- 「VPS 実行役」= VPS の `keiba-fetch-agent`（GCS のタスク表を 60 秒おきに pull、1 本ずつ実行、`MemoryMax=250M`）。
- 「GCP Jobs」= Cloud Run Jobs（Cloud Scheduler / Cloud Tasks から起動）。2026-10-28 時点で ready=true は `odds-drift-weekly`・`reserve-fetch-tasks`・`event-pump` の 3 本だけ。他は未デプロイ。

## 0. To-Be の方針

1. **netkeiba に触れる実行者は VPS 実行役 1 本に集約する。** 固定 IP で到達が実証済み、同時 1 本の制約（DEC-023）を守れる、ログインの Cookie を 1 か所に閉じられる。GCP Jobs からの到達は未検証（`architecture_target.md` 7.4）なので、試験が通るまで取得は載せない。
2. **取得を伴う工程のうち計算が重い部分は分離して GCP Jobs に回す。**
3. **ホストは学習・評価・特徴量の再構築だけを担当する。**
4. 取得結果は VPS 実行役から GCS に直接書く。ホストは学習のために GCS から読む側になる（取得系の `gcs-mirror-push` は不要になる）。
5. 最悪の保険: VPS の IP が遮断されたら、取得だけホスト（自宅 IP）に戻す。

## 1. 取得（スクレイピング）

| # | 工程 | As-Is（ホスト） | To-Be | 備考 |
|---|---|---|---|---|
| 1 | `entries-sync`（出馬表の段階収集） | 08〜22 時、10 分おき | **VPS 実行役** | 影モードの対象。軽い |
| 2 | `raceday-runner`（T-60/T-15 オッズ） | 07:30 起動の常駐ループ | **VPS 実行役**（`race-odds-per-race`） | 前夜に GCP が各 R のタスクを予約。厳守したい T-10/5/2 は期限の 60 秒前から待機 |
| 3 | `raceday-weight-runner`（馬体重の検知） | 08:00 起動の常駐ループ | **VPS 実行役**（`race-weight-per-race`）。検知後は GCP の `event-pump` から予測へ | 常駐ループは個別タスクに置換 |
| 4 | `raceday-result-runner`（T+15 の速報結果） | 07:30 起動の常駐ループ | **VPS 実行役**（新規タスク `race-result-t15` が必要） | `job_map` ではホストのまま（`host`）。移行先の設計が未了 |
| 5 | `raceday-evening`（速報・確定オッズ） | 17:30 | **VPS 実行役**（取得）。確定オッズの集計は GCP | 計算部分は 250M に収まらない可能性。メモリ実測が必要 |
| 6 | `raceday-comment`（厩舎コメント） | 07:40・20:00 | **VPS 実行役** | 軽い。`race_trainer_comment` は要ログイン |
| 7 | `jra-baba-morning`（クッション値・含水率） | 05〜08 時、10 分おき | **VPS 実行役** | JRA の PDF・ライブ。軽い |
| 8 | `daily-race-lists`（レース一覧） | 07:00・17:00 | **VPS 実行役** | 影モードの対象だが比較なし |
| 9 | `raceday-eve`（前日の出馬表・馬柱・追い切り・SmartRC） | 18:00 | **VPS 実行役**（取得、`raceday-eve-fetch`）。前計算は GCP の `eve-precompute`（#25） | 計算は分離済み。要ログイン |
| 10 | `new-horse-pedigree`（新馬の 5 世代血統） | 21:00 | **VPS 実行役** | 軽い |
| 11 | `recent-retry`（直近の未取得の取り直し） | 06:10・19:40 | **VPS 実行役** | 影の取得が「本番をスキップ」する点に注意（§5） |
| 12 | `barometer-retry` | 毎日 08:30、金曜 19:30 | **VPS 実行役** | 要ログイン |
| 13 | `shutuba-audit`（出馬表の再取得監査） | 06:40 | **VPS 実行役** | `job_map` に無い → 登録が必要 |
| 14 | `weekly-update`（確定結果・指数・バロメーター） | 金曜 18:00 | **VPS 実行役**（取得）。指数の計算は GCP Jobs | 計算込みで 250M は未測定。分離を推奨 |
| 15 | `horse-name-index`（馬名インデックス・成長曲線） | 金曜 18:30 | **GCP Jobs**（`job_map` どおり） | 取得ではなく集計 |
| 16 | `jma-weather`（気象庁の時別値） | 毎時 | **GCP Jobs**（気象庁は IP 遮断のリスクが低い） | `job_map` に無い → 登録が必要。VPS の枠を使わない |
| 17 | `backfill-2026-fast` / `backfill-2026-full` | 週 1（月曜 00:00・07:50） | **VPS 実行役**（低優先度・夜間窓） | 件数はリクエスト上限の枠内に収める |
| 18 | `backfill-all-horse` | 毎日 06:00 | **VPS 実行役**（低優先度・夜間窓） | 件数が多い。窓と上限の調整が必要 |
| 19 | `backfill-past`（過去分、毎時） | 毎時 5 分 | **VPS 実行役**（低優先度）。完了していれば廃止 | 間隔 4〜5 秒。完了状況の確認が先。`job_map` に無い |
| 20 | `backfill-2021-fast` / `backfill-2020-fast` | 火・金 02:00、水・土 03:00 | 完了済みなら**廃止**。未完なら VPS 実行役 | 完了判定が必要 |
| 21 | `horse-training-backfill`（調教の取り直し） | 月〜木 22:00 | **VPS 実行役**（低優先度・夜間窓） | 約 2,200 頭。要ログイン |
| 22 | `sporthorse-births-nightly`（誕生日） | 毎日 22:00 | **廃止**（`job_map` で `retire`） | 外部サイト・セッション依存 |
| 23 | `verify_past_values`（過去分の値の検査） | 2 時間おき | **GCP Jobs**（保存済みの値を読むだけ。取得はしない） | 読み取り専用。`job_map` に無い |
| 24 | `fetch_sire_lines`（種牡馬の「○○系」） | cron 無し。手動でホストが起動（netkeiba の血統ページ、60 秒間隔、日の総件数が上限に近いと待つ） | **VPS 実行役**（新規タスク `sire-line`、低優先度） | `job_map` に無い。系統台帳 `sire_line_map.json` の正本は GCS に置く |

## 2. 予測

| # | 工程 | As-Is | To-Be | 備考 |
|---|---|---|---|---|
| 25 | 前夜の予測保存と前計算（追走難度・想定オッズ） | `raceday-eve`（18:00）の後半 | **GCP Jobs**（`eve-precompute`、6GiB） | 取得は #9 で VPS。分離済みだが ready=false |
| 26 | 馬体重検知後の予測（本番の主経路） | `raceday-weight-runner`（08:00 起動の常駐）がホスト内で推論 | **GCP Jobs**（`predict-race`、6GiB。Cloud Tasks の予約実行） | 検知は VPS が出すイベントを `event-pump`（ready=true）が拾う。T-45 の保険つき |
| 27 | 開催日の朝のキャッシュ作成 | `serving-state-precompute`（07:35）と `profile`（08:10） | **GCP Jobs**（`day-prep`、16GiB。前者 2 本は置き換え済みで無効） | ホスト実測でピーク約 12GB・85 秒。Jobs の上限内か要確認 |
| 28 | ラップ予想 | `lap-forecast-morning`（07:30）と `eve`（19:30） | **GCP Jobs**（`lap-forecast-*`） | |
| 29 | 追走難度 v2 の前夜保存と実測突き合わせ | `tracking-v2-weekly`（月 08:30） | **GCP Jobs** | |
| 30 | 予測成績の日次集計と仮想購入の追記 | ホストの結果取得に連動 | **GCP Jobs**（`prediction-results-refresh`） | #4 の移行に依存 |

## 3. 集計（めぐ分析を含む）

| # | 工程 | As-Is | To-Be | 備考 |
|---|---|---|---|---|
| 31 | 履歴テーブルの増分更新・パース検査・ドリフト検査 | `export_tables_daily`（07:00） | **GCP Jobs**（`export-tables-daily` と `parse-drift-daily`） | 3GiB と 6GiB。分離済みだが ready=false |
| 32 | **めぐ分析の事前計算**（過去 30 日〜先 14 日） | `export_tables_daily.sh` の最後で実行 | **GCP Jobs**（新規 `race-analysis-daily`） | **`job_map` に無い。** 父の対応表（#24 の台帳）に依存。出力は GCS に直接書く |
| 33 | 騎手・調教師の統計 | `update_jockey_trainer_stats`（07:15） | **GCP Jobs**（`jockey-trainer-stats`） | |
| 34 | 履歴特徴量の蓄積とサービング用バンドル公開 | `feature-accumulate`（07:40） | **GCP Jobs**（6GiB） | ホストで 28 秒・ピーク 3.6GB |
| 35 | RB（めぐ指数の履歴）の作り直し | `rb-state-daily`（08:20） | **GCP Jobs**（`rb-state-daily`） | |
| 36 | 血統 PX のリッジ（半年ごとに前倒し） | `px-ridge-window`（月 03:30） | **GCP Jobs**（6GiB。必要なときだけ） | |
| 37 | めぐ指数の週次状態 | `hem_megu state`（日 07:30） | **GCP Jobs**（新規） | **`job_map` に無い** |
| 38 | 馬名インデックス・成長曲線 | `horse-name-index`（金 18:30） | **GCP Jobs** | #15 と同じ |
| 39 | オッズ乖離（t15/t60）の週次集計 | `odds-drift-weekly`（日 22:00） | **GCP Jobs**（ready=true） | |
| 40 | ドリフト週次監視・凍結候補の追跡 | `drift-monitor-weekly`（月 10:00）、`frozen-tracking-weekly`（月 10:30） | **GCP Jobs**（4GiB と 2GiB） | |
| 41 | 並走検証（ホストと GCP の成果物の一致） | — | **GCP Jobs**（`parity-check`、移行期間のみ） | 予測は決定性を固定したので（6520dcd7）完全一致で比べられる |

## 4. 通知・監視

| # | 工程 | As-Is | To-Be | 備考 |
|---|---|---|---|---|
| 42 | Slack の取得状況（異常 15 分おき、朝・夜のサマリー、週次の網羅） | `slack-scrape-*` | **GCP Jobs**（同名） | 異常通知の 15 分おきは Scheduler の枠を使う |
| 43 | 土日の予測成績通知 | `slack-results`（15〜21 時、20 分おき） | **GCP Jobs** | |
| 44 | ページ品質チェック | `page_quality_check`（08:00） | **GCP Jobs** | |
| 45 | VPS 死活 | — | **GCP**（`vps-watchdog`）と監視ポータルのハートビート | 監視ポータル拡張で実装済みの経路 |
| 46 | 共通ログインの月次更新 | `shared-login`（毎日起動、月初だけ更新） | **GCP Jobs** | |
| 47 | GCP 日次コスト報告 | — | **GCP Jobs**（`gcp-daily-cost-report`） | 課金ガードの補助 |

## 5. ホストに残る、または不要になるもの

| # | 工程 | As-Is | To-Be |
|---|---|---|---|
| 48 | 学習・評価・特徴量の再構築・MLflow | ホスト | **ホスト**（最終的にここだけ） |
| 49 | `gcs-mirror-push`（5 分おき）、`host-snapshot`（07:50）、`maintenance-deploy`、`server-watchdog`、`rotate-logs`、`ensure-monitor`、遠隔アクセス | ホスト | 取得・予測・集計が GCP に移れば不要。ホストの開発環境の維持だけ残す |
| 50 | `sync-pg-from-gcs`、`ensure-postgres` | ホスト | **廃止**（`job_map` で `retire`） |

## 6. 並走検証（影モード）の現状と差分チェックの設計

### 現状（2026-10-28 確認）
- 影モードの仕組みはある（`docs/operations/vps_agent.md` 5.5 章）。VPS の実行役が同じ取得を並走し、結果を `shadow/<日付>/<本番と同じパス>` に保存し、`shadow_report` が本番と比べる。**ただし有効化されていない**（`keiba-fetch-agent` は `disabled`/`inactive`）。
- `config/fetch_shadow.json`（2026-10-22 に負荷軽減で縮小）の対象は 5 タスクだけ: `entries-sync`（馬・枠・騎手の一致）、`race-odds-t60`・`race-odds-t15`（オッズ）、`race-weight-check`（馬体重）、`daily-race-lists`（比較なし）。`vps_agent.md` に書かれた `jra-baba-morning`・`recent-retry`・`barometer-retry`・直前オッズは**現在の設定では対象外**（ドキュメントが古い）。
- 1 日の上限は 800 件（見積りリクエスト数）。VPS 向けの 16 ジョブのうち、実質カバーできているのは #1・#2・#3・#8 の分だけ。

### 全工程を見るための設計
1. **全データの自動突き合わせ**: 影モードの書き込みは `shadow/<日付>/<同じパス>` に付け替えられる。この性質を使い、タスクごとの比較ルールをやめ、`shadow/<日付>/**` の全キーを本番の同じキーと比べて「本番のみ／影のみ／中身が違う」に分ける。取得時刻などの変動項目は除外し、残りの完全一致を見る。
2. **カテゴリ別のルール**: 約 50 カテゴリ（`HybridStorage.CATEGORY_MAP`）に既定ルール（変動項目以外が完全一致）を適用。オッズ・馬体重だけ、取得時刻が最も近いスナップショット同士で比べる例外にする。
3. **取りこぼしの検査**: ホストが取ったキーの一覧を基準に、影に無いものを数える。**取りこぼし 0 件**を合格条件にする。
4. **「取得済みならスキップ」への対策**: 影の取得は本番を読むので、`recent-retry` のような「未取得だけ取り直す」工程は本番にあるものをスキップして影が空になる。影の取得では「影に無ければ取る」読み方にする（取得コードの挙動に触れる変更。要確認）。
5. **対象の広げ方**: 軽いもの（#7・#10・#11・#12・#13・#16）を先に影に戻す。重いもの（#5・#9・#14・#6・#10）は VPS で取得中のメモリを実測してから。バックフィル（#17〜#21）は常時比較せず、移行時に件数・欠損を 1 回突き合わせる。
6. **メモリ**: 実行役の OOM 回数 0 と `measure_memory.sh` の実測を合格条件に足す。
7. **予測への影響の確認（任意・第 2 段階）**: 推論は決定性を固定したので（同じ入力なら同じ値）、影のオッズ・馬柱から予測を作れば入力の差が予測値のずれとして定量化できる。
8. **画面**: 日次レポート（`ops/fetch_shadow/daily/`）を監視ポータルの「パイプライン」ページに出す。
9. **負荷**: 並走中は netkeiba へのリクエストが約 2 倍になる。1 日の上限（現在 800 件）の見直しと、バックフィルを同じ時間帯に走らせないルールが要る（`docs/operations/scraping_policy.md`、DEC-023）。アクセス制限を検知したら即停止する現在の仕組みは維持する。

### 起動に必要な作業（ユーザー作業・root）
- VPS: `/etc/keiba-fetch/fetch.env` に `KEIBA_FETCH_SHADOW_ENABLED=1` と `KEIBA_FETCH_AGENT_ENABLED=1`、`sudo systemctl enable --now keiba-fetch-agent`。
- IAM: 影モードの実行役に `shadow/` と `ops/fetch_tasks_shadow/` 以下だけに書ける条件付き権限。
- GCP: `reserve-race-day` と `fetch-shadow-report` は ready=false・Scheduler 未登録（影のタスクを積む／日次集計を出すのに要る）。

## 7. 未決・懸念

- **`job_map` に未登録**: #4・#13・#16・#19・#23・#24・#32・#37。移行先を決めるには先に登録が必要。
- **メモリ（#5・#9・#14）**: 取得中のメモリが未測定。250M に収まらなければ取得と計算を分離する。
- **バックフィル（#17〜#21）**: 時間厳守の取得と衝突しないよう GCS のリースで排他し、夜間窓（JST 22〜06 時）だけ流して 1 日の上限（2,500 件）に収める。完了済みの年（#20）は廃止を検討。
- **要ログインの取得**（#6・#9・#12・#21 ほか）: 秘密は VPS だけに置く。セッションの同時使用による無効化を避けるため 1 本ずつ流す。
- **費用**: Jobs を全部動かすと、`cost_assumptions.json` の前提（月 6 万〜17 万 vCPU 秒）を超える可能性がある。重いもの（`day-prep` 16GiB、`eve-precompute`、`feature-accumulate`、`parse-drift-daily`、`px-ridge-window`、`predict-race`）の実行時間を計測してから再見積りする。前回の見積りは、全タスク込みの中央で月約 ¥1,600（幅 ¥650〜2,800）。
- **めぐ分析（#32）は #24 に依存**: 系統の取得が完全に終わるまで群を出さない仕様のため、取得を VPS に移すときは台帳の移行を先にやる。
- **GCP の IP から netkeiba に届くか**（`architecture_target.md` 7.4）は未試験。不合格なら全ての取得を VPS 実行役に集約する。

## 8. 切り離しの順番（提案）

1. #1〜#3（影モードで検証する軽い開催日タスク）。
2. #7・#8・#10〜#13（軽い日次系）。
3. #5・#6・#9・#14（メモリの実測のあと）。
4. #4（新規タスクを設計してから）。
5. #17〜#21（バックフィル）は最後。
6. 計算系（§2・§3・§4）は、GCP Jobs のデプロイ（`ENABLE_JOBS_DEPLOY`）と `parity-check` での並走検証を経て、1 本ずつホストの cron を停止する。

## 9. 網羅性の確認方法

- ホストの cron: `ssh keiba-remote 'sudo -n crontab -l -u root'`（本書は 2026-10-28 時点）。
- 登録: `python3 -c "import json; ..."` で `config/gcp_job_map.json` の `id`・`destination` を確認。
- cron に無い手動起動の取得（#24 のような）は、`scripts/` と `src/scripts/scraping/` を `netkeiba` で検索して拾う。

## 10. 導入後（To-Be）の月間コスト試算（2026-10-28）

再計算: `python -m src.scripts.ops.cost_estimate --scenario N_after_decoupling --format md`（中央は `N_after_decoupling_mid`）。前提は `config/cost_assumptions.json`。為替 1 ドル=150 円。**単価はいずれも公式ページでの確認が済んでいない**（`--list-unverified`）。実績は Jobs の稼働後に `gcp-daily-cost-report` で置き換える。

### 10.1 合計（円/月）

| | 少なめ | 中央 | 多め |
|---|---:|---:|---:|
| GCP 合計（下の内訳） | 約 625 | 約 1,740 | 約 3,670 |
| 運用コスト: ドメイン `megu-keiba.com`（年額 数千円を 12 で割る） | 約 250 | 約 330 | 約 420 |
| 運用コスト: ConoHa VPS | 0（36 か月前払い済み、2028-05-18 満了） | 0 | 0 |
| 運用コスト: Cloudflare（Tunnel・Access・Pages・R2 無料枠） | 0 | 0 | 0 |
| 運用コスト: GitHub（公開リポジトリ。Actions 無料）、Slack | 0 | 0 | 0 |
| **月間合計** | **約 900** | **約 2,100** | **約 4,100** |

- 予算は月 2,000 円（GCP + Cloudflare + ConoHa）。**中央で予算を少し超え、多めでは大きく超える**。
- 「少なめ／中央／多め」の違いは次の 3 点でほぼ決まる: ①API のコールドスタート回数（1 日 8／20／40 回）、②Jobs の実行時間（0.5 倍／1 倍／2 倍）、③Scheduler のジョブ数（6／14／35 本）。
- Jobs の無料枠（instance 課金の月 24 万 vCPU 秒など）が Jobs にも効くかは未確認。効かない最悪の場合、少なめ 約 1,140／多め 約 4,380（GCP のみ）。
- ホスト PC の電気代・回線、Claude Code の利用料は含まない（本番の実行経路に LLM の API 呼び出しは無い）。

### 10.2 GCP の内訳（円/月）

| 項目 | 少なめ | 中央 | 多め | 主な前提 |
|---|---:|---:|---:|---|
| GCS（保存・読み書き・外向き通信） | 547 | 1,169 | 1,929 | 保存 39／45／60GB。書き込み 15／40／60 万回。**読み取り 410 万／1,010 万／1,820 万回**（API の起動 1 回あたり約 13,200 件を読むのが主因）。外向き 3／6／10GB |
| Cloud Run Jobs | 0 | 約 330 | 1,150 | 下の 10.3。無料枠が効かない最悪は 340／（中央 約 680）／1,370 |
| Cloud Run API（prod＋stg、リクエスト課金） | 0 | 0 | 0 | 毎日 100 人・約 8 万リクエスト・起動 300／750／1,350 回（1 回約 50 秒）が無料枠に収まる |
| Cloud Scheduler | 45 | 165 | 480 | 6／14／35 ジョブ（無料 3）。ディスパッチャに統合するほど安い |
| Secret Manager | 18 | 36 | 54 | 有効バージョン 8／10／12（無料 6） |
| Artifact Registry | 15 | 38 | 53 | イメージ 1.5／3／4GB（無料 0.5GB） |
| Cloud Logging・Cloudflare R2 | 0 | 0 | 0 | 無料枠内 |
| 監視ポータル（keiba-monitor、リクエスト課金） | 0 | 0 | 0 | 既存の試算（M0）に含む |

### 10.3 Cloud Run Jobs の実行時間の前提（中央）

| 工程（#） | 回数/月 | 1 回 | vCPU／GiB | vCPU 秒 | GiB 秒 |
|---|---:|---:|---|---:|---:|
| `predict-race`（#26、1 レース 1 回） | 324（36R×9 開催日） | 90 秒 | 2／6 | 58,000 | 175,000 |
| `day-prep`（#27） | 9 | 120 秒 | 4／16 | 4,300 | 17,300 |
| `eve-precompute`（#25） | 18 | 300 秒 | 2／6 | 10,800 | 32,000 |
| `feature-accumulate`（#34） | 30 | 120 秒 | 2／6 | 7,200 | 21,600 |
| `export-tables-daily`（#31） | 30 | 300 秒 | 1／3 | 9,000 | 27,000 |
| `parse-drift-daily`（#31） | 30 | 300 秒 | 2／6 | 18,000 | 54,000 |
| `race-analysis-daily`（#32） | 30 | 300 秒 | 1／2 | 9,000 | 18,000 |
| `jockey-trainer-stats`・`rb-state-daily`・`lap-forecast-*`（#33・#35・#28） | 各 30〜60 | 120 秒 | 1／2 | 14,400 | 28,800 |
| 週次 8 本（#29・#36〜#40 ほか） | 約 34 | 300 秒 | 1.5／3 | 15,000 | 31,000 |
| `slack-scrape-alerts`（#42、15 分おき） | 2,880 | 15 秒 | 0.5／1 | 21,600 | 43,000 |
| その他の通知・監視（#42〜#47） | 約 700 | 15 秒 | 0.5／1 | 5,000 | 10,000 |
| `reserve-fetch-tasks`・`event-pump`（開催日の日中は毎分） | 約 5,000 | 10 秒 | 0.5／0.5〜1 | 27,000 | 21,600 |
| **合計** | | | | **約 199,000** | **約 479,000** |

- 実行時間は、ホストでの実測（`day-prep` 85 秒・`feature-accumulate` 28 秒）と、Jobs の起動・状態の取り込みの上乗せ分の**見積り**。Jobs が稼働するまで確定できない。
- 予測が全体の約 3 割を占める。1 レース 1 回の起動をやめ、開催日ごとの一括実行にすると、この分は大きく減る見込み。

### 10.4 コストを下げる手段（効果の大きい順）

1. **API の起動時の GCS 読み取りを減らす（約 −550 円/月、中央）**: 起動のたびに `TREES` の約 13,200 ファイル（152MB）を読み切る設計を、必要なものだけ先に読み、残りは表示のときに読む（LAZY）設計に寄せ、約 1,000 件に抑える。コールドスタートの回数が増えても費用が伸びにくくなる。中央が約 1,190 円（運用コスト込みで約 1,520 円）になり、予算内に収まる。
2. **Scheduler のジョブを統合する（最大 −300 円/月）**: 35 本 → 14 本以下。ディスパッチャ（`keiba-dispatch-*`）に寄せる。
3. **予測の Job を一括実行にする**: 開催日ごとに 1 回（または数回）にまとめ、起動・状態取り込みの上乗せを減らす。
4. **`slack-scrape-alerts`（15 分おき）を、監視ポータル（Cloud Run のリクエスト課金）内の処理に移す**: Jobs の起動 2,880 回/月を減らせる。
5. 完了済みのバックフィル（#20）や廃止対象（#22・#50）を実際に止める（取得は VPS なので GCP の費用は動かないが、リクエスト上限を空ける）。

### 10.5 計算に入っていないもの・注意

- 取得（スクレイピング）は VPS 実行役なので GCP の追加課金は 0。ただし VPS の月額は前払い済みのため、**満了後（2028-05）の継続費用は別途**。
- 取得を GCP Jobs に移す場合（`B_scrape_on_cloud_run_jobs`）は +最大 780 円/月。送信元 IP が固定でないため、推奨しない。
- 影モード（並走検証）の期間は、取得が約 2 倍になるが GCP の費用はほぼ増えない（VPS 側で動く）。`shadow/` 以下の保存量と、日次集計の Job 1 本が増える程度。
- Jobs の実行回数が前提を超えた場合、`gcp-daily-cost-report` と課金ガード（`api_cost_guard`）で気づく設計。Jobs は課金ガードの対象外（API のみ）なので、Jobs の費用の監視は日次報告に頼る。
