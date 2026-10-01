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

## 目標（推測）

ユーザに表示するデータの品質（欠損・不整合）を、ユーザが気づく前に運用者が検知・修復できる
状態にすること。ユーザ視点では「表示されているレース情報・指数が信頼できる」ことが最終目的で、
本ブランチ自体はそれを裏で保証する運用者向け機能と推測される。

## このラインまで実装できたらブランチを消してよい

- スキーマ検証・カバレッジチェックが自動修復トリガーと連動し、運用者が毎日手動確認しなくても
  品質劣化に気づける状態になっている
- ユーザに表示される情報が品質チェック未通過のまま出ることが無いと確認できている
- 上記が実現できていれば、ユーザ向けの実装は完了したとみなせる

## 既知の課題

- `/monitor`・開発者専用監視ポータル(:9090、`src/monitor/app.py`)はいずれもVPS上の
  常駐プロセス前提（`start_monitor.sh`でnohupデーモン化、`.monitor.pid`管理）。GCPへ移行する
  場合、Compute Engineでのリフト&シフトならそのまま常駐させられるが、Cloud Run等の
  スケールtoゼロ環境を選ぶ場合は常時起動が必要なポータルの維持方法を別途検討する必要がある
  （採用するGCPサービスにより対応が変わる）。
  詳細は [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)。
  現時点ではVPS運用継続が方針のため、本ファイルのTODOはそのまま進めてよい。

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

### 共通TODO（ホスト方式に関係ない）

- [ ] `/api/quality-check/remediate`（修復）が自動トリガーと連動しているか、現状手動実行のみかを確認する
- [ ] 品質チェック未通過のデータがユーザ表示側で実際にブロック・警告されているか確認する
      （チェック機能はあるが表示側との連動が無ければ意味が薄い）

### 常時稼働ホスト（VPS / GCP Compute Engine）の場合のTODO

- [ ] `/monitor`（レガシー版）と開発者専用監視ポータル(:9090)の役割重複を整理し、
      どちらかに統合できないか検討する（:9090ポータルは常駐プロセス前提のため、
      常時稼働ホスト方式でのみ現行の形のまま統合検討ができる）

### GCPサーバーレス（Cloud Run等）移行時のTODO

- [ ] 上記の統合検討は、サーバーレス移行する場合は先に「常時起動が必要なポータルの維持方法」
      （[`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)参照）
      を決めてからでないと着手できない

## メモ
