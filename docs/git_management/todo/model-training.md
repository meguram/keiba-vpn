# TODO: feature/model-training

**対象領域**: モデル学習・シミュレーション・バックフィル
**関連ドキュメント**: [../feature-model-training.md](../feature-model-training.md)
**最終更新**: 2026-10-01

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-10-01時点）

- モデル学習: `POST /api/train`（単体、バックグラウンド）・`/api/train/status`（状態確認、
  失敗時は`traceback`にスタックトレースを含む）
- アンサンブル学習: `POST /api/train/ensemble`（LightGBM + XGBoost + CatBoost + NN）・
  `/api/train/ensemble/status`（失敗時は`traceback`にスタックトレースを含む）
- `/api/model/info`: MLflowに登録されているモデル情報（`ModelTrainer.MODEL_NAME`=
  `"keiba-lgbm-nopi"`の最新バージョン）に加え、`used_by`（既知の利用コードパス）・
  `catalog_models`（`MODEL_CATALOG`全モデルのServing/ローカルBooster/ヒューリスティック
  対応状況）・`notes`（`keiba-lgbm`と`keiba-lgbm-nopi`の対応関係の注記）を返す
- `/api/train/compare`（新規）: 複数モデル・複数バージョンのMLflow Registry情報と、
  学習・アンサンブル・シミュレーションの現在ジョブ状態、直近バックテストの上位パラメータ
  組み合わせを並べて返す（`?models=key1,key2&versions=N`）
- バックテストシミュレーション: `POST /api/simulation/run`・`/api/simulation/status`
  （失敗時は`traceback`にスタックトレースを含む）・
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
- （2026-10-01追記）ライブAPIのレース予測応答
  （`src/pipeline/inference/race_prediction_service.build_race_prediction_response`）は
  MLflow Registryを経由せず、ローカル`models/keiba_model.pkl`（無ければヒューリスティック）を
  使用している。`src/pipeline/mlflow/catalog.py`の`keiba_lgbm`（registry名`keiba-lgbm`、
  lifecycle=active）はこの経路から未参照で、本番で実際にMLflow Registryから読み込まれている
  着順予測モデルは`"keiba-lgbm-nopi"`（`RaceDayPipeline`・`composite_optimizer`経由、
  CLI/バッチのみ）。ライブAPIの推論をcatalog/MLflow経由に統合するかどうかは別途検討が必要
  （モデリング挙動が変わるため`docs/html/modeling/`との整合確認を伴う大きめの変更になる）。

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

### 共通TODO（ホスト方式に関係ない）

- [x] 複数モデル・複数バックテスト結果を並べて比較できるUI/APIが無いため、追加を検討する
      （現状は個別ジョブのstatus確認のみ）— 2026-10-01対応: `GET /api/train/compare`
      （`?models=key1,key2&versions=N`）を新規追加。`MODEL_CATALOG`の各モデルについて
      MLflow Registry上の直近N バージョンのmetrics/paramsを並べて返し、同時に
      学習・アンサンブル学習・シミュレーションの現在ジョブ状態、および直近の
      `composite_optimizer`最適化実行内の上位パラメータ組み合わせ（`backtest_top_combinations`、
      既存の`models/composite_params.json`の`top_combinations`を再利用）も併せて返す。
      フロントエンドUIの新規画面は対象外（バックエンドAPIのみ）。複数回の最適化実行を
      時系列で比較する履歴機能（grid search単位ではなく実行単位の履歴）は未対応で、
      必要なら`composite_optimizer._save_result`に実行履歴の追記保存を加える追加TODOとして残る。
- [x] 学習・シミュレーションジョブが失敗した際に、`/status`から原因（スタックトレース等）まで
      追跡できるか確認する — 2026-10-01対応: 確認した結果、従来は`_training_job`/`_ensemble_job`/
      `_sim_job`とも`str(e)`のみを`error`に保存し、スタックトレースはサーバログにしか残らず
      （`_run_training`はログにも`exc_info`無し）、`/api/train/status`・
      `/api/train/ensemble/status`・`/api/simulation/status`のレスポンスからは原因追跡
      できなかった。`src/api/app.py`の`_run_training`・`_run_ensemble_training`・
      `_run_simulation`を修正し、例外発生時に`traceback.format_exc()`を各ジョブ辞書の
      `traceback`キーに保存、対応する3つの`/status`エンドポイントのレスポンスに
      `traceback`フィールドを追加（成功時は`None`）。既存フィールドは変更せず追加のみなので
      後方互換。
- [x] MLflowに登録されたモデルのうち、実際に本番（推論）で使われているものと学習済みモデルの
      対応関係が分かるようにする — 2026-10-01対応: 調査の結果、`src/pipeline/mlflow/runtime.py`の
      `platform_health()`/`model_health()`が`MODEL_CATALOG`全キーについて
      `infer_backend`（`mlflow_serve`/`local_mlflow`/`heuristic`）・`serve_uri`・
      `local_booster_loaded`等を既に返しており（`GET /api/inference/health`で利用中）、
      対応関係を示す仕組み自体は既に存在していた。一方`GET /api/model/info`は
      `ModelTrainer.MODEL_NAME`（`"keiba-lgbm-nopi"`、大衆指標排除モデル）固定の単一モデルのみを
      見ており、catalog.pyの`keiba_lgbm`（registry名`"keiba-lgbm"`、紛らわしいが別エンティティ）
      とは連携していなかった。さらに調査の過程で実際の不整合を発見した: ライブAPIのレース予測
      応答（`src/pipeline/inference/race_prediction_service.build_race_prediction_response`）は
      現状MLflow Registryを一切経由せず、ローカル`models/keiba_model.pkl`（無ければ
      `RaceDayPipeline._fallback_score`のヒューリスティック）を使っており、catalogの
      `keiba_lgbm`（lifecycle=active）エントリは本番推論から未参照。実際に`"keiba-lgbm-nopi"`を
      Registryから読み込んでいるのは`src/pipeline/inference/race_day.py`の
      `RaceDayPipeline._load_model`（CLI/バッチ本番推論）と`composite_optimizer.py`
      （バックテスト）のみ。対応として`GET /api/model/info`を拡張し、`used_by`
      （既知の利用コードパス一覧）・`catalog_models`（`platform_health()`の全モデル
      Serving/ローカルBooster/ヒューリスティック対応状況）・`notes`
      （上記の`keiba-lgbm`と`keiba-lgbm-nopi`の不一致、およびライブAPIがMLflowを経由していない
      事実の明記）を追加した（既存フィールドは変更なし、追加のみで後方互換）。
      ライブAPIの予測ロジック自体をcatalog経由のMLflow推論に統合する改修は、モデリング挙動の
      変更を伴う大きめの変更になるため今回は実施せず、本ファイルに新規の既知課題として記録する
      （下記「既知の課題」参照）。

（上記3件はUI/APIの機能追加・トレーサビリティの話で、ホスト方式（VPS/GCP）には依存しない。
長時間学習処理の実行基盤自体がホスト方式に依存する点は「既知の課題」を参照）

## メモ
