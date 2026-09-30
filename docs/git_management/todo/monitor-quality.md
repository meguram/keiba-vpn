# TODO: feature/monitor-quality

**対象領域**: 監視・データ品質チェック
**関連ドキュメント**: [../feature-monitor-quality.md](../feature-monitor-quality.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- `/monitor`: スクレイピング状態のリアルタイムモニタリングボード（レガシー版。開発者専用の独立監視ポータル :9090 とは別物）
- `/data-viewer`: 生JSONデータビューア
- カバレッジマトリクス: `/api/date-race-matrix`（レース×カテゴリ）・`/api/date-raw-matrix`（Raw: GCS+PG/要件）・`/api/date-calculated-matrix`（Calculated: megu_index/flat parquet）・`/api/coverage-calendar`（年別カテゴリカレンダー）
- データ品質チェック: `/api/quality-check/enqueue`（投入）・`jobs`（一覧）・`health`（健全性）・`remediate`（修復）・`calendar`（カレンダー表示）
- `/api/row-data-coverage`: 行固有派生カテゴリ（race_shutuba_meta等）のGCSカバレッジ
- `/api/monitor/missing-dates-summary`: 開催日別のJRAレース未取得件数集計
- `/api/monitor/opening-date-info`: 開催日種別（非開催ラベル用）
- `/api/monitor/context`: UI向け環境・GCS/DB接続・集計モード情報

## 既知の課題

（無し）

## TODO（手動追記用）

- [ ]

## メモ
