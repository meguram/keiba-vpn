# TODO: feature/core-platform

**対象領域**: コアプラットフォーム（ヘルス/認証/ダッシュボード）
**関連ドキュメント**: [../feature-core-platform.md](../feature-core-platform.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- `/`（ダッシュボード）: 各サービスの死活・キュー概要等を表示
- `/login`・`/logout`: Cookie ベースの認証（ログイン/ログアウト）、`/api/auth/status` でログイン状態確認
- `/api/health`: サーバー稼働状態（監視cron用）
- `/api/inference/health`: MLflow Tracking / 全モデル Serving / ローカル Booster / キャッシュの疎通確認
- `/api/gcs-stats`: GCS APIコール数・キャッシュ統計（コスト監視用）
- `/api/data/{category}/{key}`: GCS上の生JSONデータを返す汎用ビューア用エンドポイント
- `/api/html-archive/cleanup`: HTMLアーカイブのカテゴリ別世代管理（keep件数のみ残す）

## 既知の課題

（無し）

## TODO（手動追記用）

- [ ]

## メモ
