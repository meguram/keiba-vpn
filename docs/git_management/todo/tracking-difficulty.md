# TODO: feature/tracking-difficulty

**対象領域**: 追走難度
**関連ドキュメント**: [../feature-tracking-difficulty.md](../feature-tracking-difficulty.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- `/tracking-difficulty`: ページ表示
- `/api/race/{race_id}/tracking-difficulty`: 追走難度・ペース・位置取り（calculated_data 事前計算を返す。`refresh=true`で再計算）
- `POST /api/race/{race_id}/tracking-difficulty/precompute`: バッチ計算してstorageに保存（推論ワーカー相当）
- `/api/tracking-difficulty/status`: 事前計算ストアの件数・パス（読み取り専用）
- `POST /api/tracking-difficulty/train`: 追走難度モデルの学習実行
- Flask v1 (`/api/v1/races/<race_id>/tracking-difficulty` + `.../precompute`) と2026-09-30にパラメータ完全パリティ化済み
  （legacy/v1どちらも同じ`tracking_difficulty_service.get_or_compute`を利用。v1側の`HybridStorage()`直接生成も
  `_get_storage()`シングルトンに統一済み）

## 目標（推測）

ユーザが「このレースは差しが届きやすいか、逃げ有利か」を事前に把握できるようにすること。
legacy/v1どちらの画面から見ても同じ追走難度が表示される一貫性も目標に含まれると推測される。

## このラインまで実装できたらブランチを消してよい

- ユーザがどの画面（legacy/v1どちらの実装元）から見ても追走難度の値が食い違わない
- 事前計算が切れておらず、ユーザが見た際に「未計算」表示に当たる頻度が実質ゼロ
- 上記が実現できていれば、ユーザ向けの実装は完了したとみなせる（2026-09-30時点でv1パリティ・
  HybridStorageシングルトン化は解決済みのため、実質このラインに近い状態）

## 既知の課題

（無し。2026-09-30時点でv1パリティ・HybridStorageシングルトン化ともに解決済み。`make test` 439 passed で確認）

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

- [ ] legacy/v1で追走難度の値が一致することを確認する自動テストを追加する（現状は手動確認のみ）
- [ ] 事前計算（`precompute`）バッチの実行スケジュール有無を確認し、無ければ整備する
      （現状は手動トリガーのみに見える）
- [ ] 「未計算（not_precomputed）」に当たる頻度を計測する

## メモ
