# TODO: feature/race-detail

**対象領域**: レース詳細・予測表示
**関連ドキュメント**: [../feature-race-detail.md](../feature-race-detail.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- `/race/{race_id}`: レース詳細ページ表示
- `/api/race/{race_id}`: レースの全データ集約（60秒キャッシュ付き）
- `/api/race/{race_id}/bundle`: 全カテゴリ + SmartRC + 全馬horse_resultを並列GCSリードで一括取得
- 予測表示: `/api/race/{race_id}/predictions`（キャッシュ済み結果をGCSから返す、stgはモックフォールバック）。
  2026-10-01: レスポンスに `freshness`（`age_hours`/`is_stale`/`stale_threshold_hours`等。
  `_meta.scraped_at` 基準、既定しきい値24h・`KEIBA_PRE_RACE_PREDICT_ENABLED`とは別の
  `KEIBA_PREDICTION_STALE_HOURS`で変更可）を付与し、閾値超過時は`logger.warning`でログにも記録するようにした
  （`src/api/app.py` `_prediction_freshness`）。既存フロント（`templates/race/race_detail.html`）は
  未知キーを無視するため後方互換。
- 予測実行: `POST /api/race/{race_id}/predict`（開発者権限必須。AI予測を実行しGCSに保存）
- `/api/race/{race_id}/bloodline-aptitude`: 父・母父の舞台適性（dev=モック / stg=DB集計値）
- `/api/race/{race_id}/result-status`: 結果（確定・速報）の有無と結果ページURL
- 一覧系: `/api/race-list/{date}`（会場別）・`/api/upcoming-races`（未来レース）・`/api/megu-index-dates`（めぐ指数計算済み日付）
- `/api/predictions`: トップレベルのキャッシュ済み予測一覧

## 目標（推測）

ユーザが1レースについて知りたい情報（出馬表・結果・AI予測・血統適性）を1ページで確認でき、
かつAI予測がいつ見ても最新かつ一貫した内容で表示されることが目標と推測される。

## このラインまで実装できたらブランチを消してよい

- ユーザがレース詳細ページを開いたとき、予測結果が「表示されない」「古い」「見るたびに違う値になる」
  といった事象に遭遇しない
- predictions のGCS/PostgreSQL二重経路がユーザ体験に影響しない（不整合が実際に出ていない、
  または解消されている）ことが確認できている
- 上記が実現できていれば、ユーザ向けの実装は完了したとみなせる

## 既知の課題

- [ ] **predictions のGCS/PostgreSQL二重経路**: `/api/race/{race_id}/predictions`（本ブランチ、GCS）と
      Flask v1 `/api/v1/races/<race_id>/predictions`（PostgreSQL）でデータソース・レスポンス形状
      （`predictions`キー vs `horses`キー）が異なる。3系統の書き込みパス
      （推論パイプライン・legacy手動トリガ・バッチCLI）はいずれも自動実行されておらず現状は実害無しと判断し
      対応を保留中（2026-09-30ユーザ確認済み）。`pre_race_predict_trigger.py`の本実装や関連cron登録に
      着手する前に `docs/operations/service-endpoints.md`「レース予測の書き込みパスが3系統ある」を再確認すること。
      2026-10-01追記: VPS/GCP分担に伴うCloud SQL接続自体（`src/db/cloud_sql.py`・
      `src/db/session.py`の`KEIBA_DB_BACKEND=cloud_sql`分岐）は実装済み。本項目（GCS/PostgreSQL
      二重経路の整理そのもの）は未解決のまま。

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

### 共通TODO（ホスト方式に関係ない）

- [x] predictions のGCS/PostgreSQL二重経路について、`docs/operations/service-endpoints.md`の
      再検討条件（自動化着手）に該当する変更が無いか定期的に確認する（既知の課題を参照）
      — 2026-10-01対応: `crontab -l` / `/etc/cron.d/` を再確認した結果、
      `run_inference.sh`・`pre_race_predict_trigger.py`・`sync_pg_from_gcs.sh` のいずれも
      自動登録なし（2026-09-30時点と同一）。`pre_race_predict_trigger.py` も
      既定 `mock=True`・docstring「最終版は後補完」のまま変更無し。2026-09-30以降の関連コミット
      （`1ef4364` cronジョブ失敗時Slack通知追加、`b2695aa` structure_check通知修正、`a4f3be3`
      アーキテクチャヘルスチェック対応）はいずれも本件（レース予測の書き込みパス）とは無関係。
      再検討条件に該当する変更は発生していないため、変更なし・保留継続が適切と判断。
      次回確認時も本行の手順（crontab確認 + `pre_race_predict_trigger.py`の`mock`既定値確認）を踏襲すること。
- [x] `/api/race/{race_id}/predictions` が「表示されない/古い」場合の検知（モニタリング）を追加する
      — 2026-10-01対応: `src/api/app.py`の`get_race_predictions`に`_prediction_freshness()`を追加し、
      レスポンスに`freshness`（`scraped_at`/`scraped_at_jst`/`age_hours`/`is_stale`/`stale_threshold_hours`）
      を付与。`_meta.scraped_at`（`HybridStorage.save`が自動付与）基準でage_hoursを算出し、
      既定24h（`KEIBA_PREDICTION_STALE_HOURS`で変更可）を超えると`is_stale=true`かつ
      `logger.warning`でログに記録する。「表示されない」場合は既存どおり404 `{"status":"not_found"}`
      （stgはモックへフォールバック）。既存テスト（`tests/api/test_endpoints.py`含む`tests/api/`全体、
      169 passed/2 skipped）に影響なしを確認済み。
- [x] `/api/race/{race_id}/bloodline-aptitude`（dev=モック/stg=DB集計値）のstg実データ精度を検証する
      — 2026-10-01対応: ロジックレベルで検証した。集計元
      `src/scripts/maintenance/aggregate_sire_aptitude.py`の`run_aggregation()`に対し、
      finish_position・odds・place_oddsを手計算済みの合成データ（FakeStorageでrace_result/horse_result
      をモック）を`dry_run=True`で投入し、`n_runs`/`n_wins`/`n_place`/`win_odds_acc`/`place_odds_acc`
      が手計算値と完全一致することを確認（例: 対象sire 4走・2勝・4着内・win_odds_acc=5.0・
      place_odds_acc=4.7）。`win_rate = n_wins/n_runs`・`roi_win = win_odds_acc/n_runs`等の
      DB upsert側の計算式もコードレビューで追加バグ無しと確認。消費側
      `src/api/app.py`の`/api/race/{race_id}/bloodline-aptitude`（stg分岐）も
      `SireAptitudeCache`を`sire_name`/`sire_type`/`surface`/`distance_band`/`track_condition`で絞り
      `computed_at`降順最新1件を取得する設計で、集計側のキー構造と整合していることを確認した。
      なお実際のstg PostgreSQL（`sire_aptitude_cache`実データ）への接続検証は、本環境に`.env`/GCS認証情報
      および既存のPostgreSQL起動が無く実行不可のためスキップ（ロジックの妥当性確認で代替、との方針どおり）。

### VPS側（サービング）に残るTODO

（本ファイルでは該当なし。予測の実行はGCP側、配信（GCSから結果を読む）のみVPS側に残る）

### GCP側（スクレイピング・ML・スケジュール実行）のTODO

- [ ] `pre_race_predict_trigger.py`の本実装（cron登録含む）に着手する。2026-10-02決定の
      役割分担・設計方針（[`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)
      「ユーザーリクエスト起点の計算処理をどこで実行するか」参照）により、予測実行はGCP側
      Cloud Scheduler + Cloud Run Jobsで行い、結果をGCSへ書き込む設計に確定。VPS側は
      `/api/race/{race_id}/predictions`でGCSから結果を読んで配信するだけでよい（既存の
      `POST /api/race/{race_id}/predict`手動トリガーも、将来的にはVPS側からGCP側Cloud Tasksへ
      ジョブを委譲するプロキシに変更する想定）。本体の実装（`pre_race_predict_trigger.py`の
      具体的なロジック）自体は2026-10-02時点で未着手。

## メモ
