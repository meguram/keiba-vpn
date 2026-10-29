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
| 4 | `raceday-result-runner`（T+15 の速報結果） | 07:30 起動の常駐ループ | **VPS 実行役**（`race-result-t15`。コードは済み・未有効） | `job_map` ではホストのまま（`host`）。`DEFAULT_AGENT_TASKS` への追加まではホストの常駐が担当 |
| 5 | `raceday-evening`（速報・確定オッズ） | 17:30 | **VPS 実行役**（取得）。確定オッズの集計は GCP | 計算部分は 250M に収まらない可能性。メモリ実測が必要 |
| 6 | `raceday-comment`（厩舎コメント） | 07:40・20:00 | **VPS 実行役** | 軽い。`race_trainer_comment` は要ログイン |
| 7 | `jra-baba-morning`（クッション値・含水率） | 05〜08 時、10 分おき | **VPS 実行役** | JRA の PDF・ライブ。軽い |
| 8 | `daily-race-lists`（レース一覧） | 07:00・17:00 | **VPS 実行役** | 影モードの対象だが比較なし |
| 9 | `raceday-eve`（前日の出馬表・馬柱・追い切り・SmartRC） | 18:00 | **VPS 実行役**（取得、`raceday-eve-fetch`）。前計算は GCP の `eve-precompute`（#25） | 計算は分離済み。要ログイン |
| 10 | `new-horse-pedigree`（新馬の 5 世代血統） | 21:00 | **VPS 実行役** | 軽い |
| 11 | `recent-retry`（直近の未取得の取り直し） | 06:10・19:40 | **VPS 実行役** | 影の取得が「本番をスキップ」する点に注意（§5） |
| 12 | `barometer-retry` | 毎日 08:30、金曜 19:30 | **VPS 実行役** | 要ログイン |
| 13 | `shutuba-audit`（出馬表の再取得監査） | 06:40 | **VPS 実行役** | `job_map` に登録済み（2026-10-29。当面 `host`。K6 のコードは 2026-10-29 に入れた（その場で取り直す直接モード）ので、VPS の実行役が開いたら `vps` へ直す） |
| 14 | `weekly-update`（確定結果・指数・バロメーター） | 金曜 18:00 | **VPS 実行役**（取得）。指数の計算は GCP Jobs | 計算込みで 250M は未測定。分離を推奨 |
| 15 | `horse-name-index`（馬名インデックス・成長曲線） | 金曜 18:30 | **GCP Jobs**（`job_map` どおり） | 取得ではなく集計 |
| 16 | `jma-weather`（気象庁の時別値） | 毎時 | **GCP Jobs**（気象庁は IP 遮断のリスクが低い） | `job_map` に無い → 登録が必要。VPS の枠を使わない |
| 17 | `backfill-2026-fast` / `backfill-2026-full` | 週 1（月曜 00:00・07:50） | **VPS 実行役**（低優先度・夜間窓） | 件数はリクエスト上限の枠内に収める |
| 18 | `backfill-all-horse` | 毎日 06:00 | **VPS 実行役**（低優先度・夜間窓） | 件数が多い。窓と上限の調整が必要 |
| 19 | `backfill-past`（過去分、毎時） | 毎時 5 分 | **VPS 実行役**（低優先度）。完了していれば廃止 | 間隔 4〜5 秒。完了状況の確認が先。`job_map` に登録済み（2026-10-29、`host`） |
| 20 | `backfill-2021-fast` / `backfill-2020-fast` | 火・金 02:00、水・土 03:00 | 完了済みなら**廃止**。未完なら VPS 実行役 | 完了判定が必要 |
| 21 | `horse-training-backfill`（調教の取り直し） | 月〜木 22:00 | **VPS 実行役**（低優先度・夜間窓） | 約 2,200 頭。要ログイン |
| 22 | `sporthorse-births-nightly`（誕生日） | 毎日 22:00 | **廃止**（`job_map` で `retire`） | 外部サイト・セッション依存 |
| 23 | `verify_past_values`（過去分の値の検査） | 2 時間おき | **GCP Jobs**（保存済みの値を読むだけ。取得はしない） | 読み取り専用。`job_map` に無い |
| 24 | `fetch_sire_lines`（種牡馬の「○○系」） | 日々の取得は cron 不要（定期バッチの5世代血統の取得に相伴して、台帳に無い祖先の系統を取る）。`fetch_sire_lines` は過去分の一括補完用で手動起動（60 秒間隔、日の総件数が上限に近いと待つ） | **VPS 実行役**（新規タスク `sire-line`＝GCS との台帳の同期だけ。通信は GCS のみ。実際の系統取得は定期バッチに相伴） | `job_map` に登録済み（2026-10-29、`host`。タスクのコードも 10/29 に入れた）。系統台帳 `sire_line_map.json` の正本は GCS に置く。**並走中の扱い**: 取得はホスト（定期バッチ・一括）と VPS/GCP が同時に走ってよい。台帳は書き込みをロックして読み直すので同じ機械の複数プロセスで取りこぼさず、別の機械の台帳は `python -m src.scraper.sire_line_store <相手のjson>` で差を見て、`--merge` で足せる（種牡馬 ID が鍵で、表記ありが『表記なし』に勝ち、順序によらず同じ結果）。**取得の失敗の扱い**: ページは正常で系統の表記が無い種牡馬は、理由（`no_line`/`no_father`/`father_mismatch`）つきで `line=null` として台帳に残し、取り直さない（merge でも表記ありが勝つ）。通信エラーや血統表の無い応答は、3 秒・10 秒の間を置いて最大 3 回取り直し、それでも駄目なら台帳に残さず（後の馬の取得で再挑戦）、一括の側は完了の印を付けない。アクセス制限の疑い（HTTP 400/403 の連続）は取り直さず止める。VPS 側は空の台帳から始めず、先にホストの台帳を写してから走らせる（でないと取得済みの約 4,500 頭を取り直す） |

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
| 37 | めぐ指数の週次状態 | `hem_megu state`（月 07:30） | **GCP Jobs**（新規） | **`job_map` に無い** |
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
- 影モードの仕組みはある（`docs/operations/vps_agent.md` 5.5 章）。VPS の実行役が同じ取得を並走し、結果を `shadow/<日付>/<本番と同じパス>` に保存し、`shadow_report` が本番と比べる。**VPS の `keiba-fetch-agent` は影モードで稼働中**（リリース `fetch-0063931f`。新しいコードは 03:20 JST の取り込みで入る）。`fetch_owner.json` の `agent_tasks` は空なので、本番の取得は引き続きホストの cron が行う。
- `config/fetch_shadow.json`（2026-10-22 に負荷軽減で縮小）の対象は 5 タスクだけ: `entries-sync`（馬・枠・騎手の一致）、`race-odds-t60`・`race-odds-t15`（オッズ）、`race-weight-check`（馬体重）、`daily-race-lists`（比較なし）。`vps_agent.md` に書かれた `jra-baba-morning`・`recent-retry`・`barometer-retry`・直前オッズは**現在の設定では対象外**（ドキュメントが古い）。
- 1 日の上限は 800 件（見積りリクエスト数）。VPS 向けの 16 ジョブのうち、実質カバーできているのは #1・#2・#3・#8 の分だけ。

### 全工程を見るための設計
1. **全データの自動突き合わせ**: 影モードの書き込みは `shadow/<日付>/<同じパス>` に付け替えられる。この性質を使い、タスクごとの比較ルールをやめ、`shadow/<日付>/**` の全キーを本番の同じキーと比べて「本番のみ／影のみ／中身が違う」に分ける。取得時刻などの変動項目は除外し、残りの完全一致を見る。
2. **カテゴリ別のルール**: 約 50 カテゴリ（`HybridStorage.CATEGORY_MAP`）に既定ルール（変動項目以外が完全一致）を適用。オッズ・馬体重だけ、取得時刻が最も近いスナップショット同士で比べる例外にする。
3. **取りこぼしの検査**: ホストが取ったキーの一覧を基準に、影に無いものを数える。**取りこぼし 0 件**を合格条件にする。
4. **「取得済みならスキップ」への対策**: 影の取得は本番を読むので、`recent-retry` のような「未取得だけ取り直す」工程は本番にあるものをスキップして影が空になる。影の取得では「影に無ければ取る」読み方にする。**実装済み**（`src/ops/shadow_storage.py` の `install()` が `HybridStorage.exists` / `exists_gcs` を影の側だけ見るように差し替える。`load` などの読みは本番のまま、他の操作は従来どおり拒否）。影の側に取りに行く分だけ、影モードのリクエスト数（1 日 800 件の上限）の消費は増える。
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

- **`job_map` の登録**: 2026-10-29 に #13・#19・#24 を追加して、棚卸しの全工程が登録済みになった（#4・#23・#32・#37 は 10/28 の時点で登録済みだった。#16 は表の備考のとおり）。#13・#19・#24 は K6 のコードが無いので、移行先を `vps` にするとテスト（`fetch_task.task` が許可リストの中）が通らない。**`destination=host` で登録した。K6 のコードは 10/29 に入れたので、VPS の実行役（K3）が開き、ホストの台帳を GCS へ写してから `vps` へ直す。**（#13・#24 の `vps` 化は、テストが通るかどうかではなく運用の順番で止めている）
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

1. **【実装済み 2026-10-29・デプロイ後に実測】API の起動時の GCS 読み取りを減らす（約 −550 円/月、中央）**: 起動のたびに `TREES` の約 13,200 ファイル（152MB）を読み切る設計を、必要なものだけ先に読み、残りは表示のときに読む（LAZY）設計に寄せ、約 1,000 件に抑える。コールドスタートの回数が増えても費用が伸びにくくなる。中央が約 1,190 円（運用コスト込みで約 1,520 円）になり、予算内に収まる。
2. **Scheduler のジョブを統合する（最大 −300 円/月）**: 35 本 → 14 本以下。ディスパッチャ（`keiba-dispatch-*`）に寄せる。
3. **予測の Job を一括実行にする**: 開催日ごとに 1 回（または数回）にまとめ、起動・状態取り込みの上乗せを減らす。
4. **`slack-scrape-alerts`（15 分おき）を、監視ポータル（Cloud Run のリクエスト課金）内の処理に移す**: Jobs の起動 2,880 回/月を減らせる。
5. 完了済みのバックフィル（#20）や廃止対象（#22・#50）を実際に止める（取得は VPS なので GCP の費用は動かないが、リクエスト上限を空ける）。

### 10.5 計算に入っていないもの・注意

- 取得（スクレイピング）は VPS 実行役なので GCP の追加課金は 0。ただし VPS の月額は前払い済みのため、**満了後（2028-05）の継続費用は別途**。
- 取得を GCP Jobs に移す場合（`B_scrape_on_cloud_run_jobs`）は +最大 780 円/月。送信元 IP が固定でないため、推奨しない。
- 影モード（並走検証）の期間は、取得が約 2 倍になるが GCP の費用はほぼ増えない（VPS 側で動く）。`shadow/` 以下の保存量と、日次集計の Job 1 本が増える程度。
- Jobs の実行回数が前提を超えた場合、`gcp-daily-cost-report` と課金ガード（`api_cost_guard`）で気づく設計。Jobs は課金ガードの対象外（API のみ）なので、Jobs の費用の監視は日次報告に頼る。

## 11. 実施戦略（2026-10-28）

### 11.1 考え方

- **50 工程は 50 個の作業ではない。** 実際に手を動かす山は、下の「鍵」の 6 つだけ。鍵が開けば、残りは各工程で同じ手順の繰り返しになる。手順は「Jobs で 1 回手動実行 → 並走で一致を確認 → `ready=true` → 合否表で合格 → ホストの cron を 1 行止める → 14 日観察」。判定は `cron_cutover_status` が機械的に出す。
- **律速はカレンダー。** 合格には開催日 3〜5 日分の並走（`parity_min_days`・`raceday_min_days`・影モード 5 日）が要る。開催日は土日だけなので、**並走の時計を一日でも早く動かす**ことを最優先にする。コードの準備はその間に進める。
- **4 本の線で並行して進める。** A 計算（GCP Jobs）、B 取得（VPS 実行役）、C 通知・監視（GCP Jobs）、D 後片付け（ホストの常駐の停止）。A と B は互いを待たない。C は A の成果物、D は A〜C の全部を待つ。
- 基準点（2026-10-28 の `cron_cutover_status`）: 合格 0、不合格 13、未着手 34、N/A 14、無効 14、対象外 2。

### 11.2 鍵（これが開かないと先に進めない作業）

| 鍵 | 内容 | 開くもの | 担当 |
|---|---|---|---|
| K1 | **Jobs のイメージを最新にする**（master への昇格で `ENABLE_JOBS_DEPLOY=true`）。今の `jobs:fefbef34` は sec77 より古く、想定オッズが出ない | A・C の全工程の手動実行（未着手 34 本の大半） | **済（2026-10-29）**。担当はユーザー（昇格の承認）。詳細は §12.2 |
| K2 | **state の種まき**（`state_sync push --which in`）: `tables`・`horse_history`・`serving_bundle`・`inference/state` | `predict-race`・`day-prep`・`eve-precompute` ほか、state を読む全工程 | Claude（ホストから GCS の `state/` だけに書く。運用のパスには書かない） |
| K3 | **VPS の取得の実行役と影モードを有効にする**（`fetch.env` の 2 変数、`systemctl enable --now keiba-fetch-agent`）と、影モード用の条件付き IAM | B の全工程（不合格 13 本の大半は「影モードの日次レポートが 0/5 日」） | **済（2026-10-29）**。担当はユーザー（VPS の root と IAM）。詳細は §12.2 |
| K4 | `reserve-race-day` と `fetch-shadow-report` を ready にする | 影のタスクの予約と日次の合否 | Claude（K1 の後） |
| K5 | 秘密の登録（共通ログイン・Slack Webhook を Secret Manager に）と、BigQuery への課金エクスポート | `shared-login`・通知系・`gcp-daily-cost-report` | ユーザー（Console） |
| K6 | 新しい取得タスクのコード: `race-result-t15`（#4・実装済み、`DEFAULT_AGENT_TASKS` には未登録）、`shutuba-audit`（#13・実装済み）、`sire-line`（#24・実装済み、系統一括取得の完了後に切替）、`backfill-past`（#19・未着手）。影の取得の「影に無ければ取る」読み方（§6 の 4）は実装済み | B の残り（#4・#11・#13・#19・#24） | Claude（コード。取得の挙動に触れるので、差分は都度確認をもらう） |

### 11.3 線ごとの順番

**A 計算（GCP Jobs）** — state の流れの上流から順に開ける。ほぼ全部が `data/page_reference/tables` を読むので、根は `export-tables-daily`。
1. `export-tables-daily` → `feature-accumulate`・`parse-drift-daily`（g-0700 の直列）
2. `rb-state-daily`・`jockey-trainer-stats`・`lap-forecast-*`（tables だけを読む。並べて開けられる）
3. `day-prep`（16GiB。ホストの実測ピークは 12GB）
4. `eve-precompute` → **`predict-race`**（D-02 の本体。予測は決定性を固定済みなので、`parity-check` で完全一致を見る）
5. `prediction-results-refresh` → `slack-results`（#4 の結果取得が VPS に移ってから本番化）

**B 取得（VPS 実行役）** — §8 の順番どおり。K3 を開けた週末から影モードの時計を動かす。
1. #1〜#3・#8（影の対象。比較ルールあり）
2. #7・#10〜#13（軽い。#13 は K6 の後）
3. #5・#6・#9・#14（VPS で取得中のメモリを測ってから。250M を超えたら、取得と計算を分ける）
4. #4（K6 の `race-result-t15`）
5. #17〜#21（バックフィル。最後）

**C 通知・監視** — A の成果物が GCS に出てから。`slack-scrape-*` は、元データ（`scrape_*`）を VPS 実行役の結果記録に置き換える設計が先。

**D 後片付け** — 最後にまとめて行う。
- **`gcs-mirror-push` は一番最後に止める。** 画面が読む `ops/mirror/` は、今はホストが 5 分おきに上げている。止めてよいのは、この木に書く工程（予測・出馬表の状態ほか）が**すべて** GCP・VPS の `mirror_out` に移ってからだけ。並走中の Jobs は `KEIBA_PARALLEL_RUN=1` で `mirror_out` を書かないので、二重書きは起きない。
- その後、`host-snapshot`（並走の比較用）、`server-watchdog`、トンネル、`ensure-monitor` を止める。完了の判定は `decoupling_plan.md` 6 章の試験（ホストの API・トンネル・cron を全部止めても公開が続く）。

### 11.4 カレンダー（目安。開催日の並走の日数で決まる）

| 期間 | A 計算 | B 取得 | 担当がユーザーの作業 |
|---|---|---|---|
| 10/28〜10/30 | K2 の種まき。A-1・A-2 の準備（job_map の不足分の登録） | K6 のコード着手 | K1・K3 は済（10/29）。残りは K5（Console） |
| 10/31〜11/1（開催） | 並走 1〜2 日目（`predict-race` は手動で数レース） | 影 1〜2 日目（#1〜#3・#8） | — |
| 11/2〜11/6 | A-1〜A-3 を 1 本ずつ手動実行して ready に | 影を #7・#10〜#12 に広げる。メモリの実測 | K5 |
| 11/7〜11/8（開催） | 並走 3〜4 日目 → `predict-race` の合否が出る | 影 3〜4 日目 | — |
| 11/9〜11/15 | 合格した工程から、ホストの cron を 1 行ずつ停止 | 影 5 日目以降 → #1〜#3・#8 の担当を VPS に移す | cron 停止の承認 |
| 11/16〜11 月末 | C を開ける | #5・#6・#9・#14、#4 | — |
| 12 月 | D 後片付け。バックフィルの移行 | — | ホストの常駐の停止 |

### 11.5 守ること

- **ホストの cron は、合否表で合格した単位（`cut_unit`）だけを、1 行ずつ止める。** 止める前の crontab は `data/local/meta/cron_backups/` に残し、戻すときはその行を戻すだけにする。止めた後も 14 日は観察する（`observe_days_after_stop`）。
- **ホストで運用の工程を直接動かさない。** 2026-10-27 に、ホストで `predict_race` を `KEIBA_ENV=stg` のまま動かして、本番 GCS の想定オッズを上書きした。比較は `host_snapshot` と `parity-check` の正式な経路だけで行う。
- 並走中の netkeiba へのリクエストは約 2 倍になる。1 日の上限（800 件）を超えないよう、影モードの期間はバックフィルを夜間窓（22〜06 時）に限る。アクセス制限を検知したら即停止する仕組みは維持する。
- 費用は、Jobs を開けるたびに `gcp-daily-cost-report`（K5 の後）で実績を見る。中央の見積りで予算を少し超えるので、§10.4 の 1（API の起動時の GCS 読み取りを減らす）を A と並行して進める。

### 11.6 棚卸しの補正（2026-10-28 に確認）

- #37 `hem_megu state` の As-Is は**月曜 07:30 JST**（cron は UTC の日曜 22:30）。表は修正済み（元は「日 07:30」）。
- #4 `raceday-result-runner` は `job_map` に登録済み（`destination=host`）。足りないのは移行先のタスク（K6）。
- #19 `backfill-past` は**まだ終わっていない**。2019 年は 3,452 レースで終わり、2018 年は 2,519 レースで途中、2016・2017 年は 0 件。#23 `verify_past_values` はこの補填の検査なので、#19 と一緒に扱う（補填が終わったら両方とも廃止）。
- #32 めぐ分析の事前計算は、`export_tables_daily.sh` の最後で `precompute_race_analysis` として動いている。GCP では g-0700 の直列の最後に置く（Scheduler のグループは予算の上限 14 本に達しているので、新しいグループは作らない）。

### 11.7 移行と一緒に入れる最適化（メモリ・費用・効率）

方針: 工程を GCP・VPS に移すときに、**同じ処理をそのまま載せ替えず、読む量・書く量・起動回数を減らしてから載せる**。移した後に直すと、並走の検証をやり直すことになるため。

| 状態 | 対象 | 内容 | 効果 |
|---|---|---|---|
| 済（10/28） | `state_sync push` | GCS に同じ中身（サイズと md5 が一致）があるファイルは上げない | 予測系の種まきの上げ直しが 262 ファイル・431MB → 0 ファイル（7 秒）。Jobs の `state_out` の書き戻しも同じ扱い |
| 済（10/28） | `state_sync pull` | サイズをキーごとに問い合わせず、一覧 1 回で取る | Jobs の起動ごとの GCS 操作が「ファイル数 × 2 回」減る（予測系は 1 回の起動で約 520 回） |
| 済（10/28） | `jockey-trainer-stats`・`race-analysis-daily` | tables 全部（2.9GB、上限超過で動かない設定）を、実際に読む `*/race_result_flat.parquet`（22MB）に絞る | 取り込みが約 130 分の 1。メモリの跳ね上がりの元を除く |
| 済（10/29・実測待ち） | API の起動時の取り込み（§10.4 の 1） | 実装は下の §12.1 のとおり。元の方針: 起動時に約 13,000 ファイルを全部降ろすのをやめる。件数の 9 割は `predictions_backfill`（5,530）・`lap_forecast`（2,776）・`tracking_v2`（2,424）・`race_lists`（1,477）で、どれもレース・日付ごとに 1 ファイル。**起動時は直近（例: 前後 14 日）だけ降ろし、古いものは読まれたときに 1 件だけ取る**（既存の `fetch_one` と `LAZY_TREES` の仕組みを使う）。読む側が `glob` で探している箇所（`local_predictions.py` の `*/{日}/{レース}.parquet` など）は、GCS の一覧で場所を引く形に直す | 中央の見積りで月 約 −550 円（予算内に収まる）。起動も速くなる |
| 次 | `predict-race`（§10.4 の 3） | 1 レース 1 起動をやめ、同じ時刻帯のレースを 1 回の起動でまとめて予測する（state の取り込み 430MB を 1 回で済ませる） | Jobs の費用の約 3 割の部分が大きく減る。並走の検証の前に入れる |
| 次 | `day-prep`（16GiB、ピーク 12GB） | 読む列を絞る・年ごとに読んで集計する形に直し、8GiB 以下を目標にする | Jobs のサイズを xl → l に下げられる |
| 後 | `slack-scrape-alerts`（15 分おき）（§10.4 の 4） | 監視ポータル（リクエスト課金）の中の処理に移す | Jobs の起動が月 2,880 回減る |
| 後 | Scheduler（§10.4 の 2） | 新しい工程は既存のグループに載せる（今日の 4 本もそうした）。上限 14 本を守る | 最大 月 −300 円 |

## 12. 進捗（2026-10-29）

### 12.1 済んだこと

| 鍵・項目 | 状態 | 内容 |
|---|---|---|
| K2 state の種まき | **済** | `state_sync push --which in` を実行。GCS の `state/` だけに書き、運用のパスには書いていない。中身は約 26MB で、GCS の保管・操作の費用は無視できる。上げ直しは、サイズと md5 が同じものを飛ばすので 0 件 |
| job_map の不足分 | **済** | #13 `shutuba-audit`・#19 `backfill-past`・#24 `sire-line` を登録（`destination=host`、`reason` つき）。`tests/ops/test_gcp_job_map.py` は 17 件とも通る。これで棚卸しの全工程が `job_map` にある |
| API 起動時の GCS 読み取り削減（§10.4 の 1） | **コード済み・develop** | `predictions_backfill`・`lap_forecast`・`tracking_v2` の日付つきキーのうち、既定で 14 日より古いもの（環境変数 `KEIBA_MIRROR_WINDOW_DAYS` で変更）は起動時・更新時に降ろさない。日付の無いファイル（`_shadow_models.json` など）は常に降ろす。古い日は読まれたときに `lazy_fetch` で 1 件だけ取る（窓の内側は従来どおり取り込み済みとみなし、無駄な GCS 問い合わせをしない）。読む側 5 箇所（`local_predictions.py` の 2 関数・`lap_forecast_service.load_forecast`・`tracking_v2_service.load_saved`・`serving_v6.stage_forecast_columns`）を `lazy_fetch` 経由にした。Jobs の `pull_trees`・CLI の pull/push は変更なし（全件を扱う）。テスト 4 件を追加、全体 3,714 件が通る。効果（約 13,200 → 約 1,000 件）はデプロイ後の起動ログで実測する |
| K6（`shutuba-audit`・`sire-line`） | **コード済み・develop** | 上の §12.3 のとおり。`AUTO_SCRAPE_MODES` と `auto_scrape.TASKS` に登録、`DEFAULT_AGENT_TASKS` には未追加（費用・メモリ未実測）。テスト 8 件（直接モード 4 件＝キューを使わない・1 回の上限・試行回数・アクセス制限、台帳の GCS 同期 4 件）を追加し、`tests/ops`・`tests/scripts` を含め 354 件が通る。取得の挙動に触れるのは `shutuba_audit` の直接モードだけ（既定オフ、`fetch_child` 経由のみオン） |
| #24 系統の取得 | 進行中 | 5 世代血統表の牡の祖先の系統（3,699 頭）を、ホストで 5 秒間隔で一括取得中（2026-10-29 開始、約 7〜8 時間）。日々の取得は定期バッチに相伴するので個別 cron は不要。台帳は並走（ホスト＋VPS/GCP）に耐える作り（ロック・merge・diff）。失敗は『系統なし（理由つき）』と『通信エラー（3 回取り直し）』を分けた |

### 12.2 まだのこと（担当つき）

| 鍵 | 担当 | 状態 |
|---|---|---|
| K1 master への昇格（Jobs のイメージ更新） | ユーザー | **済（2026-10-29）**。master `c7f8bd47`、promote run 40820261726 が全ステップ成功。Run Job 4 本が `jobs:c7f8bd47`。1 回目は Scheduler 更新の `--headers` で落ち、`--update-headers` に直して再昇格した。A・C の手動実行はこれの後 |
| K3 VPS の実行役の有効化と条件付き IAM | ユーザー | **済（2026-10-29）**。VPS の実行役は影モードで安定稼働（13:23 UTC から）。GCP 側は Run Job 4 本に `KEIBA_FETCH_SHADOW_ENABLED=1` が入った。実際の影タスク（`shadow_tasks` > 0）は未来のレース日（土曜 10/31）の `reserve-race-day` 後に確認 |
| K5 秘密の登録・BigQuery の課金エクスポート | ユーザー | 秘密は 10/24 に登録済み（`SLACK_WEBHOOK_URL_PROD`/`_STG`/`R2_CREDENTIALS` は任意）。課金エクスポートは **済（2026-10-29）**（`billing_export`、asia-northeast1、`keiba-jobs` に読み取り権限）。残りは GitHub のリポジトリ変数 `GCP_BILLING_BQ_TABLE` の設定と昇格（`promote.yml` が Jobs の環境変数に渡す）。その後 `gcp-daily-cost-report` を ready にする |
| K4 `reserve-race-day`・`fetch-shadow-report` | Claude | K1・K3 が済んだので着手可。`fetch-shadow-report` は ready で g-0730 に登録済み。影モードはホストの系統一括取得と並走中で、netkeiba へのリクエストが約 2 倍になる点に注意。2026-10-29 に `reservations race-day --date 20261031 --dry-run`（影モード有効）で `shadow_tasks`=106（18 レース）を確認。実際の積みは 10/30 前夜の reserve-race-day、初回の合否は 10/31 夜〜11/1 07:30 の fetch-shadow-report |
| K6 新しい取得タスクのコード | Claude | `shutuba-audit`・`sire-line` はコード済み（§12.1）。`race-result-t15`（#4）もコード済み・未コミット・未有効（§12.3）。常駐ループの後処理は GCP の既存 Job へ移す実装を入れた（馬場速度は `export-tables-daily`、集計は `prediction-results-refresh`。§12.3）。残りは、影を「無ければ取る」読み方にする件。`backfill-past`（#19）は移行しない方針。差分はユーザー確認のため提示する |
| 系統台帳の GCS への初回アップロード | Claude（ホスト経由） | ホストの一括取得（`complete_at` が付く）が終わった後に、`python -m src.scraper.sire_line_store` 経由の `sync_with_gcs` を 1 回走らせる。VPS の `sire-line` はその後 |

### 12.3 K6 の設計メモ（`shutuba-audit`・`sire-line` はコード済み）

- `shutuba-audit`（#13）: 今の `run_audit(enqueue=True)` は実行機のローカルのキューに積むだけで、VPS にはそのキューを消化するものが無い。VPS では積まず、**古い出馬表のレースをその場で直接再取得**する形（`_run_race_tasks_direct(runner, race_id, date, ["race_shutuba"])`）にする。試行回数は今と同じ `shutuba_audit_state.json`（上限 3 回）で数える。**実装**: `run_audit(direct=True)` が、1 回あたり最大 12 レース（`DIRECT_MAX_PER_RUN`）を `_run_race_tasks_direct` で取り直す。試行回数は取る前に数えて保存する（途中で落ちても無限に取り直さない）。アクセス制限の応答が出たら残りは取らずに止める。残りは次の回に回す。`fetch_child` が `shutuba-audit` のとき `KEIBA_SHUTUBA_AUDIT_DIRECT=1` を既定にする（ホストの今の cron は従来のキュー積みのまま）。取得に失敗したらタスクの状態を `error` にして `FETCH_RESULT` の終了コード 3 で返す。
- `sire-line`（#24）: `fetch_sire_lines` の中身（`fetch_and_record`）を取得タスクにする。**実装は同期だけ**: `sire_line_store.sync_with_gcs()` が `ops/sire_lines/sire_line_map.json` を読んで台帳に足し合わせ（表記ありが『表記なし』に勝つ）、足した結果を GCS へ戻す。`complete_at` はどちらかにあれば両方へ付く。GCS が壊れた JSON のときは取り込まずに上書きする。実際の系統取得は定期バッチの5世代血統の取得に相伴する（ユーザー確認済み）ので、一括取得スクリプトの `PED_DIR` に依存する掃引は VPS に持ち込まない。先にホストの台帳を GCS へ写してから VPS で走らせる（でないと取得済みの約 4,500 頭を取り直す）。
- `backfill-past`（#19）: 補填が終わる（2018 年が途中、2016・2017 年が 0 件）までホストで続ける。終われば #23 と一緒に廃止するので、VPS 移行はしない方針。
- `race-result-t15`（#4）: **コード済み（2026-10-29）**。`fetch_plan day` が発走 15 分後に 1 レース 1 件を積み（`--no-result` で外せる）、`fetch_child` が `race_result_on_time` を 1 レース分だけ直接取得する（タイムアウト 180 秒）。`fetch_owner` により、`agent_tasks` に入るとホストの `raceday-result-runner` は見送る。影モードの対象（`config/fetch_shadow.json`）には入れていない（負荷を増やさない）。
  - **常駐ループの後処理の移し先（2026-10-29 実装）**: VPS のタスクは取得だけにする（250MB 枠で、flat と馬場速度の正本は GCS のため）。新しい Job・Scheduler は作らない（§12.4）。
    - 馬場速度: 既存 Job `export-tables-daily`（07:00 JST、flat の更新の直後）に手順 `src.scripts.data.track_speed_daily --from {days_ago_10} --to {today}` を足した（`allow_fail`）。`races_{year}.parquet` へのアップサートなので、直近 10 日を毎朝計算し直せば、月曜夕方に出る確定結果も翌朝に拾える。基準表（`track_speed_baselines.parquet`・`track_speed_pace_baselines.parquet`、計 約31KB）は静的なので state_in に足しただけ。出力 `data/page_reference/track_speed`（約1.2MB）は state_out。
    - `prediction_results` の集計: 既存 Job `prediction-results-refresh`（`g-results`）が担当済み。
    - `recent-retry`: `recent_retry.run` は `KEIBA_RUN_LOCATION=vps` のとき後処理を呼ばず、`post_processing: "gcp"` を返す（ホストでは従来どおり後処理する）。
    - `race-result-t15` を `DEFAULT_AGENT_TASKS` に入れる前の残り: ①`export-tables-daily` を Jobs で 1 回手動実行し、ホストと馬場速度が一致することの確認（`ready=false` のまま）、②`prediction-results-refresh` の前提（ホストの過去予測の GCS 反映・`predict-race` の `predictions_v6`）、③VPS のメモリ実測。
- いずれも `AUTO_SCRAPE_MODES` に足して `auto_scrape.TASKS` に対応する関数を持たせる（`tests/ops/test_fetch_task.py` が両者の整合を見る）。`DEFAULT_AGENT_TASKS` への追加は、VPS でメモリを実測した後（`shutuba-audit`・`sire-line` もまだ入れていない）。

### 12.4 費用を上げないための決まり（この移行の間は必ず守る）

- **GCP に新しい Cloud Run Job・Cloud Scheduler・min-instances を作らない。** 新しい工程は既存の Scheduler グループ（上限 14 本）に載せる。
- **取得は VPS かホストで行う**（GCP の追加課金は 0）。取得を Jobs に移すと最大 +780 円/月になるので移さない（§10.5）。
- **API を 15 分以内の間隔で呼ぶものを作らない。** API はインスタンス課金で、15 分以内の ping が続くと月約 7,800 円になる。ハートビート・集計・点検の受け先は `keiba-monitor` に置く。
- K2 の種まきは GCS の約 26MB の保管のみ（無視できる）。`state_sync` は同じ中身を上げ直さない。
- Jobs は `api_cost_guard` の対象外なので、Jobs を ready にするたびに `gcp-daily-cost-report`（K5 の後）で実績を見る。それまでは、Jobs の手動実行を 1 本ずつ・回数を絞って行う。
- 予算内に収める本命は §10.4 の 1（API の起動時の GCS 読み取りを約 13,200 → 約 1,000 件へ、約 −550 円/月）。これは 10/29 にコードを入れた（§12.1）。デプロイ後に起動時の読み取り件数を実測し、約 1,000 件になっていることを確かめてから Jobs を開ける。
