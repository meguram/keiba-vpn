# TODO: feature/model-training

**対象領域**: モデル学習・シミュレーション・バックフィル
**関連ドキュメント**: [../feature-model-training.md](../feature-model-training.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- モデル学習: `POST /api/train`（単体、バックグラウンド）・`/api/train/status`（状態確認）
- アンサンブル学習: `POST /api/train/ensemble`（LightGBM + XGBoost + CatBoost + NN）・`/api/train/ensemble/status`
- `/api/model/info`: MLflowに登録されているモデル情報
- バックテストシミュレーション: `POST /api/simulation/run`・`/api/simulation/status`・
  `/api/simulation/params`（composite scoreパラメータ・最適化結果詳細）
- Backfill: `/api/backfill/status`・`POST /api/backfill/start`（過去データ取得）
- race_listsバックフィル: `/api/race-lists-backfill/status`・`POST .../start`・`POST .../stop`
  （`scripts/data/backfill_race_lists_kaisai_since_2020.py`をsubprocess起動・PID管理でSIGTERM停止）
- ページURLは無し（`/api/*`のみで構成される機能領域）

## 既知の課題

（無し）

## TODO（手動追記用）

- [ ]

## メモ
