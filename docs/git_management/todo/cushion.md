# TODO: feature/cushion

**対象領域**: クッション値（トラックコンディション）
**関連ドキュメント**: [../feature-cushion.md](../feature-cushion.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- `/api/cushion/data`・`/api/cushion/stats`: クッション値・含水率データの取得・統計
- `POST /api/cushion/scrape` + `/scrape/status`: スクレイピング実行・状態確認
- `POST /api/cushion/live` + `/live/status` + `/live/check`（軽量更新チェック）: JRA公式馬場情報ページからのライブ取得
- `/api/cushion/schedule`: 直近のポーリングスケジュール
- `POST /api/cushion/admin/sync-gcs`: ローカルcushion_values.jsonを年別にGCSへ同期（開発者ログイン必須）
- `POST /api/cushion/admin/sync-preprocessed`: GCS preprocessed/cushion_dataの日次JSONをjra_cushion年別JSONにマージ（開発者のみ）
- ページURLは無し（`/api/*`のみで構成される機能領域。2026-09-30にユーザ判断で本ブランチは保持継続）

## 既知の課題

（無し。以前 `/api/cushion/scrape`(+status) が `scraping-queue` に誤分類されていたが移動済み。
v1側との重複実装や`HybridStorage()`直接生成も無いことを確認済み）

## TODO（手動追記用）

- [ ]

## メモ
