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

## 目標（推測）

ここでの「ユーザ」は開発者（モデル改善に取り組む人）。新しい特徴量・アルゴリズムを試して
学習→バックテスト→比較までを画面操作で完結でき、コマンドライン直叩きが不要になることが
目標と推測される。最終的にはユーザ（エンドユーザー）が受け取る予測精度の向上に繋がる。

## このラインまで実装できたらブランチを消してよい

- 開発者が新モデルの学習・アンサンブル・バックテストを本APIから実行し、結果比較まで
  完結できている（手元スクリプトへの依存が実質不要）
- 学習・シミュレーションジョブが失敗した際に原因が`/status`系から追跡できる
- 上記が実現できていれば、（開発者という）ユーザ向けの実装は完了したとみなせる

## 既知の課題

- 学習・アンサンブル・バックテストはいずれもバックグラウンドの長時間処理（VPS上の常駐
  プロセス内スレッドやsubprocessとして実行、タイムアウト制約なし）。GCPへ移行する場合、
  Compute Engineでのリフト&シフトならタイムアウト制約を気にせず現行方式を継続できるが、
  Cloud Run等を選ぶ場合はサービスのリクエストタイムアウト制約があるため、長時間バッチは
  Cloud Run JobsやVertex AIのような基盤に分離する必要がある（採用するGCPサービスにより
  対応が変わる）。詳細は
  [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)。
  現時点ではVPS運用継続が方針のため、本ファイルのTODOはそのまま進めてよい。

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

### 共通TODO（ホスト方式に関係ない）

- [ ] 複数モデル・複数バックテスト結果を並べて比較できるUI/APIが無いため、追加を検討する
      （現状は個別ジョブのstatus確認のみ）
- [ ] 学習・シミュレーションジョブが失敗した際に、`/status`から原因（スタックトレース等）まで
      追跡できるか確認する
- [ ] MLflowに登録されたモデルのうち、実際に本番（推論）で使われているものと学習済みモデルの
      対応関係が分かるようにする

（上記3件はUI/APIの機能追加・トレーサビリティの話で、ホスト方式（VPS/GCP）には依存しない。
長時間学習処理の実行基盤自体がホスト方式に依存する点は「既知の課題」を参照）

## メモ
