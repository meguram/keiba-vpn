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
- 予測表示: `/api/race/{race_id}/predictions`（キャッシュ済み結果をGCSから返す、stgはモックフォールバック）
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

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

### 共通TODO（ホスト方式に関係ない）

- [ ] predictions のGCS/PostgreSQL二重経路について、`docs/operations/service-endpoints.md`の
      再検討条件（自動化着手）に該当する変更が無いか定期的に確認する（既知の課題を参照）
- [ ] `/api/race/{race_id}/predictions` が「表示されない/古い」場合の検知（モニタリング）を追加する
- [ ] `/api/race/{race_id}/bloodline-aptitude`（dev=モック/stg=DB集計値）のstg実データ精度を検証する

### 常時稼働ホスト（VPS / GCP Compute Engine）の場合のTODO

- [ ] `pre_race_predict_trigger.py`の本実装（cron登録含む）に着手する場合、VPS継続か
      GCP Compute Engineかに関わらずOS crontabのまま実装できる

### GCPサーバーレス（Cloud Run等）移行時のTODO

- [ ] `pre_race_predict_trigger.py`の本実装にCloud Run等のサーバーレス構成を選ぶ場合は
      Cloud Scheduler+Cloud Run Jobsでの実装が前提になる。着手前にVPS継続かGCP移行か、
      移行する場合はどちらのコンピュート方式かの方針を確認すること。詳細は
      [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)

## メモ
