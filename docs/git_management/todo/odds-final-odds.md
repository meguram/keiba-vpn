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

- [ ] 想定オッズ予測の精度を実測し、`feature/betting`のEV計算に使える精度水準かを検証する
      （現状、精度の定量評価が見当たらない）

### 常時稼働ホスト（VPS / GCP Compute Engine）の場合のTODO

- [ ] `/api/odds/snapshot/{race_id}` の記録頻度・カバレッジ（主要レースで欠落が無いか）を、
      現行のOS crontab/daemon thread方式を前提に確認する
- [ ] オッズ予測モデルの再学習サイクル（`/api/odds/train`）が定期実行されているか、
      現行方式を前提に確認する

### GCPサーバーレス（Cloud Run等）移行時のTODO

- [ ] 上記2件（snapshot記録・再学習サイクル）は、サーバーレス移行する場合はCloud Scheduler+
      Cloud Run Jobsの実行ログを前提にした確認に置き換わる。詳細は
      [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)

## メモ
