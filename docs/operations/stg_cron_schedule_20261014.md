# stg（ホスト PC）のスクレイピング cron: 時間割と取得データの網羅状況（2026-10-14）

ホスト PC の root の crontab（UTC。以下は JST に換算）。定義の正は `scripts/cron/setup_all_cron.sh`（`install` で再登録）。グループ別の一覧は `docs/html/operations/cron_schedule.html`（`python -m src.scripts.docs.build_cron_schedule_html` で再生成）。バックアップは `data/local/meta/cron_backups/`。
cron デーモンは稼働中（2026-10-13 23:44 UTC 起動）。ホストの再起動で止まった時間帯のジョブは実行されない点に注意（10/14 07:30 JST の runner は欠落、非開催日のため影響なし）。

## 1. 時間割（JST）

| 時刻 | ジョブ | 内容 |
|---|---|---|
| 毎日 00:00 | PG 同期、backfill 2026（fast） | GCS→PostgreSQL 増分、直近 7 開催日の結果・出馬表 |
| 毎日 01:00 / 02:00 / 03:00 / 04:00 | backfill 2025 / 2024 / 2023 / 2022（fast）、火金 02:00 に 2021、水土 03:00 に 2020 | 過去年の取りこぼし補填（日付ごとに 1 回で「完了」扱い） |
| 毎日 04:30 | ログローテーション | |
| 毎日 05:00〜08:50（10 分ごと） | jra-baba-morning | JRA 馬場（クッション値・含水率）。開催日のみ |
| 毎日 05:30 | 騎手・調教師統計 | |
| 毎日 06:00 | backfill horse | 馬情報一括 |
| 毎日 06:10 **（新）** | recent-retry | 直近 21 日の未取得データの取り直し |
| 毎日 07:00 | daily-race-lists（朝） | 今日以降の全開催日のレース一覧 |
| 毎日 07:00 **（新）** | export_tables_daily | 当年の flat テーブルを増分更新（推論の履歴） |
| 毎日 07:30 | raceday-runner、raceday-result-runner | 開催日のみ常駐。各R T-60 オッズ、T-15 バンドル、T+15 速報結果 |
| 毎日 07:50 / 08:00 / 09:00 | backfill full 2026 / 2025 / 2024 | 補助データ（速報結果など） |
| 毎日 08:00 **（新）** | raceday-weight-runner | 各R 発走 65〜30 分前に出馬表を取り直し、馬体重が揃ったら推論モードで特徴量を作成（開催日のみ） |
| 毎日 08:00 | ページ品質チェック | |
| 毎日 08:30 | barometer-retry | 公開済み・未取得のバロメーター |
| 毎日 17:00 | daily-race-lists（夕） | |
| 毎日 17:30 | raceday-evening | 速報結果・確定オッズ（開催日のみ） |
| 毎日 18:00 | raceday-eve | 翌日が開催日なら出馬表・馬柱・追い切り・SmartRC |
| 毎日 19:40 **（新）** | recent-retry | 同上（2 回目） |
| 金 18:00 / 18:30 / 19:30 | weekly-update / horse-name-index / barometer-retry | 直近 10 日の確定結果・指数・バロメーター、馬名インデックス |
| 常時 | watchdog（3 分ごと）、リモートアクセス（2 分ごと） | |

**ガード（新）**: daily-race-lists・weekly-update・barometer-retry・recent-retry・backfill の全行は `scripts/cron/yield_to_rescrape.sh` 経由。優先処理（`run_priority_rescrape.sh`）が動いている間は見送り（`logs/yield_skips.log` に記録）、終われば自動で通常どおり動く。netkeiba は同一アカウントで別セッションがログインすると取り直し側のセッションが無効になるため。止めたいときは `data/local/meta/rescrape/YIELD_DISABLED` を置く。raceday 系・JRA 馬場・export は見送らない。

## 2. 取得したい生データの網羅状況

「直近の開催日の取得済み / 全レース」は 2026-10-14 の実測（GCS）。

| 生データ | 用途 | 担当ジョブ | 10/3・4 | 10/10 | 10/11 | 10/12 | 評価 |
|---|---|---|---|---|---|---|---|
| 出馬表 `race_shutuba` | 前日の tier 0、馬体重（tier 2） | eve、T-15、馬体重検知（新） | 24/24 | **18/24** | 24/24 | 24/24 | 10/10 の 6R 欠け（未調査） |
| 馬柱・追い切り・オッズ・馬連系 | 特徴量・期待値 | eve、T-60/T-15、evening | 揃う | 揃う | 揃う | 揃う | ○ |
| 速報結果 `race_result_on_time` | 履歴の補修・同日の馬場差 | result-runner、evening | 揃う | 揃う | 揃う | 揃う | ○ |
| **確定結果 `race_result`** | 履歴の主材料（着順・時計・上り・通過順） | weekly（金）、backfill、**recent-retry（新）** | 24/24 | **0/24** | **0/24** | **0/24** | **穴だった。recent-retry で解消** |
| 結果ラップ `race_result_lap` | 馬別ラップ（2024-01 以降のみ存在） | **recent-retry（新）** | **0** | 0 | 0 | 0 | 穴だった（定期経路なし）。recent-retry で解消 |
| タイム指数 `race_index` | 母表の has_speed_index | weekly、recent-retry、取り直し（2020〜2025） | 24/24 | 24/24 | 24/24 | **0/24** | recent-retry で解消 |
| 厩舎コメント | 追加特徴量候補 | recent-retry（新） | 24/24 | 0/24 | 0/24 | 0/24 | 穴だった。recent-retry で解消 |
| パドック | 追加特徴量候補（上位 5 頭のみ） | backfill full、予測パイプライン | 24/24 | 24/24 | 24/24 | 0/24 | △ 定期経路が backfill のみ |
| バロメーター | 追加特徴量候補 | weekly、barometer-retry | 23/24 | 公開前 | 公開前 | 公開前 | ○（翌週金曜に公開） |
| 馬ページ・血統 | 馬の属性 | weekly、backfill horse、手動（5 世代） | – | – | – | – | △ 5 世代血統は手動 |
| JRA 馬場 | tier 1 | jra-baba-morning | – | – | – | – | ○ |

### 見つかった問題と対応

1. **最新週末の確定結果が取れていなかった（重大）**: backfill は 10/12 0:00 JST に 10/11 分を取りに行ったが、db.netkeiba にまだ結果が無く（`0 entries`）、それでも台帳に「完了」と記録したため、以降一度も再試行されなかった。金曜の weekly-update まで確定結果が入らず、土曜の推論の履歴が最大 5 日古くなる。→ `recent-retry`（新）が、直近 21 日の未取得のレース × カテゴリだけを 1 日 2 回、間隔 3 時間・最大 30 回まで試す。初回の対象は 336 件（確定結果 72、結果ラップ 168、タイム指数 24、厩舎コメント 72）。
2. **取り直しとの衝突**: daily-race-lists が、キューに溜まった 2026 年分の日付ジョブをすべて処理しており（`_kick_and_wait` が最長 48 時間キュー全体の完了を待つ）、4 時間以上走ってログインを繰り返していた。→ 上記のガードで、優先処理の実行中は見送る。**既に走っていた 1 回（17:00 JST 起動）は止めていない。**
3. 推論用の履歴テーブル（`race_result_flat` の 2026 年分）が更新されていなかった（最終日 2026-10-04）。→ `export_tables_daily`（新）を 07:00 JST に追加。
4. 馬体重の検知ランナー（新）を 08:00 JST に追加。mode は `features`（特徴量の作成まで。予測は起動しない）。

### 未解決・要確認

- 10/10 の出馬表 6R 欠け、10/12 のパドック未取得の原因。
- db.netkeiba の確定結果が何時ごろから取れるか（recent-retry の 06:10 / 19:40 で取れた時刻が `data/local/meta/recent_retry_state.json` と `logs/recent_retry.log` に残る）。取れる時刻が分かれば、1 日 2 回の時刻を詰める。
- キューに残る 2026 年分の日付ジョブ（race_list 24・date_all 5）が優先処理の後に一気に流れる。10/16 18:00 JST の raceday-eve（翌日の出馬表・馬柱・追い切り）がキューの後ろに並ばないよう、10/16 昼までにキューが空になっているかを確認する。
- 優先処理が終わった後の 2026 年分の取得（`scrape_from_health_plan`）が、ログインを使うジョブとまた重ならないか。
