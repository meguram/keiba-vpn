# TODO: feature/odds-final-odds

**対象領域**: オッズ・最終オッズ
**関連ドキュメント**: [../feature-odds-final-odds.md](../feature-odds-final-odds.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- `/api/race/{race_id}/final-odds`: 想定オッズ予測（storageキャッシュ優先、`refresh=true`で再計算）
- `POST /api/race/{race_id}/final-odds/precompute`: バッチ計算してstorageに保存（推論ワーカー相当）
- `POST /api/odds/train` + `/api/odds/train/status`: オッズ予測モデルの学習実行・状態確認
- `POST /api/odds/snapshot/{race_id}`: 指定レースの現在オッズを取得し推移履歴に記録
- `/api/odds/history/{race_id}`: オッズ推移履歴の取得
- `/api/odds/predict/{race_id}`: 予測オッズの取得
- ページURLは無し（`/api/*`のみで構成される機能領域。2026-09-30にユーザ判断で本ブランチは保持継続）

## 既知の課題

（無し。以前 `.../final-odds` が `race-detail` に誤分類されていたが移動済み。v1側との重複実装や
`HybridStorage()`直接生成も無いことを確認済み）

## TODO（手動追記用）

- [ ]

## メモ
