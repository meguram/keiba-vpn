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

## 目標（推測）

ユーザが最終オッズを事前に見越して馬券のEV（期待値）判断ができるよう、確定前オッズの精度の高い
予測を提供することが目標と推測される。

## このラインまで実装できたらブランチを消してよい

- 想定オッズの予測精度が、馬券戦略（`feature/betting`）側のEV計算に実用的な精度で使える
  レベルに達している
- オッズ推移履歴・スナップショットが主要レースで欠落なく記録されている
- 上記が実現できていれば、ユーザ向けの実装は完了したとみなせる

## 既知の課題

（無し。以前 `.../final-odds` が `race-detail` に誤分類されていたが移動済み。v1側との重複実装や
`HybridStorage()`直接生成も無いことを確認済み）
- 下記TODOの「snapshot記録頻度」「オッズ予測モデルの再学習サイクル」の確認は、VPSなら既存の
  OS crontab/daemon thread方式を前提に調査すればよい。GCPへ移行する場合も、Compute Engineでの
  リフト&シフトなら同方式を前提に調査できるが、Cloud Run等のサーバーレス構成を選ぶ場合は
  Cloud Scheduler+Cloud Run Jobsの実行ログに置き換わる（採用するGCPサービスにより対応が変わる）。
  詳細は [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)。
  現時点ではVPS運用継続が方針のため、本ファイルのTODOはそのまま進めてよい。

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

### 共通TODO（ホスト方式に関係ない）

- [x] 想定オッズ予測の精度を実測し、`feature/betting`のEV計算に使える精度水準かを検証する
      （現状、精度の定量評価が見当たらない）
      — 2026-10-01対応: 本環境では GCS が VPC Service Controls のポリシーで403拒否され
      （`storage.googleapis.com`への実アクセスで `Request is prohibited by organization's policy`）、
      ローカルにも `race_shutuba`/`race_result`/`race_odds` 等のキャッシュが無く（起動中の実サーバ
      `:8000` 経由でも同症状を確認）、過去レースの確定オッズに対する独立したMAE・相関の再実測は
      本環境では実行不可能だった。代わりに (1) 既存の学習済みバンドル
      `models/final_odds_bundle.json`（`trained_at=2026-05-21T14:30:57`）に記録済みの保留データ精度
      ＝単勝 MAPE 49.2%・RMSE(log)=0.4708、複勝下限 MAPE 48.1%、複勝上限 MAPE 58.6%
      （ただし `train_years=eval_years=["2025"]` で同年内チェーン末尾15%の内部ホールドアウトであり、
      `src/scripts/data/train_final_odds_model.py` が想定する既定の学習2020-2024/評価2025の
      年次分割では検証されていない）、(2) `feature/betting`のEV計算実体の配線確認、の2点から結論。
      配線確認の結果、`src/pipeline/inference/betting.py`の`BettingOptimizer`
      （`POST /api/betting/optimize`）はEV計算の単勝/複勝オッズを`final_odds_predictor`からではなく
      ライブ取得の`race_odds`（無ければその場スクレイプ）または`race_shutuba`+`race_odds`の
      フォールバックから取得しており、`src/pipeline/models/final_odds_predictor.py`を一切参照して
      いない。並行する価値ベット判定（`src/pipeline/inference/inference_pipeline.py`の
      `is_value_bet`/`calculate_recovery_rate`）も、着順予測モデルの勝率から導いた理論オッズ
      （`1/win_prob`）を使っており、同様に`final_odds_predictor`とは無関係。つまり想定オッズ予測は
      現状、`feature/betting`のEV計算パスに全く接続されておらず（`想定オッズ`表示列・
      `/api/race/{race_id}/final-odds`・Flask v1 `/final-odds` のための表示専用）、「使えるか」という
      問いは実装上まだ該当しない。かつ既存の保留データ精度（単勝オッズで平均約49%の誤差）は、
      `min_ev=1.05`のような閾値判定を容易に覆す規模であり、将来EV計算に接続する場合でも
      現状の精度のままでは実用に耐えない。結論: 現時点ではEV計算に未接続のため精度要件自体が
      該当せず、接続する場合は既定の年次分割（学習2020-2024/評価2025・複数年）で再評価し精度を
      改善してからにすべき。モデルの再学習は本対応では行っていない。

### VPS側（サービング）に残るTODO

（本ファイルでは該当なし。train・snapshot記録の実行はいずれもGCP側へ移動）

### GCP側（スクレイピング・ML・スケジュール実行）のTODO

- [ ] `/api/odds/snapshot/{race_id}`（`src/api/app.py:8758` `record_odds_snapshot`）の記録頻度・
      カバレッジを確認する。他ジョブ（#1〜#10）と異なり「1日1回のバッチ」ではなく
      レース発走前に複数回・レースごとに呼ぶ必要がある処理のため、単純なCloud Scheduler
      1件には落とし込めない。既存のraceday-runner/raceday-eve系cronジョブ（発走時刻スナップショット
      `race_day_schedule`を参照）と同じ発走時刻ベースのスケジューリングパターンを使う設計が必要
      （2026-10-02時点で未着手。既存raceday系cronのGCP移行と合わせて設計すること）。
- [x] snapshot記録（`/api/odds/snapshot/{race_id}`）は、サーバーレス移行する場合はCloud
      Scheduler+Cloud Run Jobsの実行ログを前提にした確認に置き換わる。詳細は
      [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)
      — 2026-10-02: 上記の通り、実行コマンドの単純な移行では済まないことが判明したため
      設計課題として残す（このTODO自体は「サーバーレス移行時の前提」の確認として完了）。
- [x] 再学習サイクル（`/api/odds/train`）は、サーバーレス移行する場合はCloud Scheduler+
      Cloud Run Jobsの実行ログを前提にした確認に置き換わる — 2026-10-01対応: `/api/odds/train`
      が呼ぶ`FinalOddsTrainer.train()`と同じ処理を行う既存CLI
      `python -m src.scripts.data.train_final_odds_model`（新規ファイル追加は不要）を確認し、
      実行コマンド・想定頻度（元はUI手動トリガーのみで固定cron無し。提案値: 毎週日曜03:00 JST）・
      リソース目安・`gcloud scheduler jobs create http`登録コマンド例を
      [`docs/operations/gcp-cloud-run-jobs.md`](../../operations/gcp-cloud-run-jobs.md)に
      まとめた。デプロイ設計図は
      [`scripts/gcp/deploy_cloud_run_jobs.sh`](../../../scripts/gcp/deploy_cloud_run_jobs.sh)。
      実デプロイ・スケジューラ登録はユーザー側作業として残る

## メモ
