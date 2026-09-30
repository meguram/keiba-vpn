# TODO: feature/growth-curve

**対象領域**: 成長曲線
**関連ドキュメント**: [../feature-growth-curve.md](../feature-growth-curve.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- `/growth-curve`: ページ表示
- `/api/growth-curve/{horse_id}`: 馬の成長曲線データ（calculated_dataローカル優先、計算成功時は随時蓄積）。
  通常GETはローカルJSONのみ（GCS読み取りなし）。`allow_compute=true`で未計算時に1回だけ計算、
  `force_refresh=true`で再計算、`fetch_speed_index=true`でrace_index補完（GCS増）、
  `jra_only`で中央競馬/全会場切替、`limit`で件数制限
- `/api/growth-curve/status`: 成長曲線ローカルストアの件数・パス
- Flask v1 (`/api/v1/horse/<horse_id>/growth-curve`) と2026-09-30にパラメータ完全パリティ化済み
  （legacy/v1どちらも同じ`growth_curve_service.get_growth_curve`を利用。v1側の`HybridStorage()`直接生成も
  `_get_storage()`シングルトンに統一済み）

## 既知の課題

（無し。2026-09-30時点でv1パリティ・HybridStorageシングルトン化ともに解決済み。`make test` 439 passed で確認）

## TODO（手動追記用）

- [ ]

## メモ
