# TODO: feature/track-speed

**対象領域**: トラックスピード指標
**関連ドキュメント**: [../feature-track-speed.md](../feature-track-speed.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- ページ: `/track-speed`・`/track-speed/dev`（開発用、`/login`必須のリダイレクト付き）
- `/api/track-speed/meta`: baseline/pace_baseline生成状態、日付数、会場一覧
- `/api/track-speed/dates`（会場フィルタ可）・`/api/track-speed/venues`（日付指定）: 集計済み日付・会場の一覧
- `/api/track-speed/day`: 指定日・会場のトラックスピードデータ取得（`track_speed_engine.query_day`）
- `/api/track-speed/status`: baseline再構築ジョブの進行状況・ready状態
- `POST /api/track-speed/rebuild-baselines`: ベースライン再構築をバックグラウンドスレッドで開始
- `POST /api/track-speed/assign`: 指定期間のレースにperf_index割当を実行
- `/api/track-speed/validate-perf`: レースパフォーマンス指数のバリデーション実行
- `/api/track-speed/race-horses`: レースの全馬にperf_indexと速度水準ラベルを付与（2着馬がrace_perfと同値）
- `/api/track-speed/by-category`: カテゴリ別集計

## 既知の課題

（無し）

## TODO（手動追記用）

- [ ]

## メモ
