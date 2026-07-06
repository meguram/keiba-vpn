# AREA-04 — 運用最適化要件（Cron SLA / プロセス分離 / Circuit Breaker / 監視・アラート / デプロイ / ロールバック）
**Status**: FINAL | **Last Updated**: 2026-07-06 | **Consolidates**: DEC-001（統合済み）

---

## 1. 概要

本仕様書は keiba-vpn プロジェクトの運用最適化要件を定義する。Cron ジョブ SLA・プロセス分離・Circuit Breaker・監視・アラート・デプロイ・ロールバックを対象とする。

---

## 2. Cron スケジュール・SLA（実装準拠）

システムタイムゾーン: UTC。cron 記述は UTC。実行スクリプトは `TZ=Asia/Tokyo` を付与して JST で動作。
実装: `scripts/cron/setup_all_cron.sh`（マスター） + `src/scraper/auto_scrape.py`（タスク定義）

### 2-0. スクレイピング対象カテゴリ × タスク対応表

| ストレージカテゴリ | 書込タスク | タイミング（JST） | 備考 |
|---|---|---|---|
| `race_lists` | `daily-race-lists` | 07:00・17:00 毎日 | 今日〜14日先の全開催日 |
| `jra_cushion` | `jra-baba-morning` | 05:00-08:50 毎10分（開催日のみ） | クッション値・含水率 |
| `race_shutuba` | `raceday-eve`（主）・`raceday-runner`（FB） | 18:00 前日 / T-15分 | 出馬表（枠順・騎手確定版） |
| `race_shutuba_past` | `raceday-eve` | 18:00 前日 | 馬柱（過去5走） |
| `race_oikiri` | `raceday-eve`・`backfill-full` | 18:00 前日 / 深夜 | 追い切りタイム |
| `horse_training` | `raceday-eve` | 18:00 前日 | 調教師コメント |
| `smartrc_race` | `raceday-eve`・`raceday-runner`・`raceday-evening`・`weekly-update` | 多段 | SmartRC 指数 |
| `race_detail` | `raceday-runner`（T-15） | T-15分 各R | 出走確定情報 |
| `race_odds` | `raceday-runner`（T-15）・`backfill-full` | T-15分 / 深夜 | 単勝・馬連オッズ |
| `race_paddock` | `raceday-runner`（T-15） | T-15分 各R | パドックコメント |
| `race_barometer` | `raceday-runner`（T-15）・`backfill-full` | T-15分 / 深夜 | レースバロメーター |
| `race_trainer_comment` | `raceday-runner`（T-15） | T-15分 各R | 調教師コメント（当日） |
| `race_result_on_time` | `raceday-result-runner`（T+15） | T+15分 各R | 速報結果 |
| `race_result` | `raceday-evening`・`weekly-update`・`backfill-fast` | 17:30 / 17:30金 / 深夜 | 確定着順・タイム |
| `race_pair_odds` | `raceday-evening`・`weekly-update` | 17:30 | 馬連・3連複確定オッズ |
| `race_index` | `raceday-evening`・`weekly-update`・`backfill-full` | 17:30 / 17:30金 / 深夜 | 速度指数・偏差値 |
| `horse_result` | `weekly-update`・`backfill-horse` | 17:30 金曜 / 06:00 毎日 | 馬個別戦績 |
| `horse_name_index` | `horse-name-index` | 18:00 金曜 | 馬名→horse_id マッピング |
| `growth_curve` | `horse-name-index` | 18:00 金曜 | 成長曲線（calculated_data） |
| 騎手・調教師統計 | `update_jt_stats` | 05:30 毎日 | `data/features/jockey_trainer_stats/` |

---

### 2-1. 常時監視

| SLA | cron（UTC） | JST | タスク | 内容 |
|---|---|---|---|---|
| Watchdog | `*/3 * * * *` | 常時 3分ごと | `server_watchdog` | API + MLflow プロセス死活監視・自動再起動。`@reboot` にも登録（起動後 15秒待機） |
| 構造監視 | 毎日 JST 06:00 | 06:00 | `structure-scheduler` | ページ構造変更検知、`versions.json` 更新 |
| ログ保全 | `30 19 * * *` | 04:30 | `rotate_logs` | ログローテーション。JST 12:00 に保全期間超ファイル削除 |
| キャッシュ保守 | 86,400 秒ごと（起動直後+以降毎日） | — | `disk_cache_cleanup` | `data/cache/` の古ファイル削除（週次アクセス < 2回は対象） |
| キュー保守 | 3,600 秒ごと | — | `queue_hourly_maintain` | 失敗ジョブを待機状態に戻し・完了レコード削除 |

### 2-2. 定期取得（毎日）

| SLA | cron（UTC） | JST | タスク名 | 実行内容 | 非開催日動作 |
|---|---|---|---|---|---|
| SLA 0a | `0 22 * * *` | 07:00 | `daily-race-lists` | 今日〜末尾の全開催日 race_lists 取得・更新 | 実行（全曜日） |
| SLA 0b | `0 8 * * *` | 17:00 | `daily-race-lists` | 同上（夕方更新） | 実行（全曜日） |
| JT統計 | `30 20 * * *` | 05:30 | `update_jt_stats` | 騎手・調教師統計再生成 | 実行（全曜日） |
| SLA 1 | `0 9 * * *` | 18:00 | `raceday-eve` | **翌日が開催日のみ**: race_shutuba + race_shutuba_past + race_oikiri + horse_training + smartrc_race → 追走難度・最終オッズ precompute | 即終了 |
| SLA 2 | `*/10 20-23 * * *` | 05:00-08:50 毎10分 | `jra-baba-morning` | jra_cushion（クッション値・含水率）ポーリング | 開催日のみ実取得 |

### 2-3. 開催日リアルタイム

| SLA | cron（UTC） | JST | タスク名 | 実行内容 |
|---|---|---|---|---|
| SLA 3 | `30 22 * * *` | 07:30 | `raceday-runner` | 開催日常駐。各R 発走 **T-15分** に T-15バンドル（race_detail + race_odds + race_paddock + race_barometer + race_trainer_comment + smartrc_race + JRA馬場ライブ）→ AI 予測トリガ |
| SLA 4 | `30 22 * * *` | 07:30 | `raceday-result-runner` | 開催日常駐。各R 発走 **T+15分** に race_result_on_time（速報結果）取得 |
| SLA 5 | `30 8 * * *` | 17:30 | `raceday-evening` | 確定結果（race_result）+ race_pair_odds + race_index → 馬場速度指数計算トリガ |

### 2-4. 週次

| SLA | cron（UTC） | JST | タスク名 | 実行内容 |
|---|---|---|---|---|
| SLA 6 | `30 8 * * 5` | 17:30 金曜 | `weekly-update` | 先週全開催日: `race_result`（確定）・`race_pair_odds`・`race_index`・`smartrc`・`horse_result` 一括更新 → 指数/偏差値/成績集計再計算 |
| — | `0 9 * * 5` | 18:00 金曜 | `horse-name-index` | 馬名リスト（`horse_name_index`）+ 成長曲線（`growth_curve`）→ `calculated_data` 一括更新 |

### 2-5. バックフィル（夜間 / 年度別）

| cron（UTC） | JST | 対象年 | フェーズ | 最大件数 |
|---|---|---|---|---|
| `0 15 * * *` | 00:00 | 2026 | fast（race_result + race_shutuba） | 7日分 |
| `0 16 * * *` | 01:00 | 2025 | fast | 5日分 |
| `0 17 * * *` | 02:00 | 2024 | fast | 5日分 |
| `0 18 * * *` | 03:00 | 2023 | fast | 5日分 |
| `0 19 * * *` | 04:00 | 2022 | fast | 5日分 |
| `0 21 * * *` | 06:00 | 全年 | horse（`horse_result` 一括） | 一括 |
| `30 22 * * *` | 07:30 | 2026 | full（`race_index`・`race_odds`・`race_barometer`・`race_oikiri` 等補助データ） | 5日分 |
| `0 23 * * *` | 08:00 | 2025 | full（同上） | 3日分 |
| `0 0 * * *` | 09:00 | 2024 | full（同上） | 3日分 |
| `0 17 * * 1,4` | 02:00 月木 | 2021 | fast（週2回） | 5日分 |
| `0 18 * * 2,5` | 03:00 火金 | 2020 | fast（週2回） | 5日分 |

---

## 3. プロセス分離

| プロセス種別 | 役割 | 備考 |
|---|---|---|
| スクレイパープロセス | netkeiba.com / smartrc.jp データ収集 | グローバル最大同時 4スロット、バースト制限付き |
| スナップショット集計バッチ | `*_stats_snapshot` 生成 | `as_of_race_id` 紐付け、results 収集完了後に起動 |
| 追走難度 precompute | `tracking_difficulty` キャッシュ事前計算 | raceday-eve 完了後に起動（`KEIBA_EVE_PRECOMPUTE_TRACKING=1`） |
| 最終オッズ precompute | `final_odds_prediction` キャッシュ事前計算 | raceday-eve 完了後に起動（`KEIBA_EVE_PRECOMPUTE_FINAL_ODDS=1`） |
| AI 予測トリガ | T-15バンドル完了後に Stage 1→2 推論実行 | `KEIBA_PRE_RACE_PREDICT_ENABLED=1` 時に起動 |
| 推論バッチプロセス | Stage 1 → Stage 2 順次推論・結果書込 | 発走 3 時間前までに完了（N-9） |
| API サーバー（Flask） | REST API 提供・Redis キャッシュ参照 | キャッシュヒット ≤ 200 ms（N-1）、ミス ≤ 2,000 ms（N-2） |
| DDL マイグレーションプロセス | Alembic スキーマバージョン管理 | デプロイ時に独立実行（N-11） |

**シングル IP 環境制約**: netkeiba.com スクレイパーはグローバルスロット 4 以内を厳守し、IP ブロックを防止する。

---

## 4. VPS メモリバジェット

DEC-001 には VPS のメモリ上限や具体的なメモリ割り当て数値の明示的な記述は存在しない。

> **要対応**: 後続の DEC（運用インフラ決定文書）で以下を確定する必要がある。
> - VPS スペック（RAM 上限）
> - 各プロセス（API サーバー・推論バッチ・Redis・PostgreSQL）へのメモリ上限割り当て
> - LightGBM / LSTM モデルのロード時メモリ見積もり

---

## 5. Circuit Breaker

DEC-001 には Circuit Breaker パターンの明示的な記述は存在しないが、以下のリトライ・バックオフ設定がその代替機能を一部担っている。

### 5-1. スクレイパーのリトライ制御

```
netkeiba.com:
  リクエスト間隔: 2.2〜4.0 秒（ランダム + ガウスジッター）
  バースト制限: 14 req ごとに 6〜12 秒クールダウン
  セッションクールダウン: 60 req ごとに 22〜40 秒
  セッションリフレッシュ: 150 req ごとに TLS/Cookie 再構築
  グローバル最大同時スロット: 4
  UA ローテーション: Chrome/Firefox/Edge × Windows/Mac/Linux 8種
  429/503 バックオフ: 初期 5s・係数 2.5・最大 3 リトライ
  403: UA 即時ローテーション後リトライ
  日次上限: 5,000 req / セッション上限: 500 req

SmartRC:
  リクエスト間隔: 2.0〜5.0 秒
  セッション上限: 200 req / 日次上限: 1,000 req
  クールダウン: 60 秒
  リトライ: 最大 3回、係数 2.0、対象: [429, 503]
  robots.txt 準拠、ブロック検知時に即停止
```

### 5-2. 結果スクレイパーのリトライ

```yaml
results:
  trigger: "発走予定時刻 + 35分"
  retry: "5分間隔 × 最大6回"  # 合計最大 30 分間リトライ
```

### 5-3. 要対応事項

> 後続 DEC で以下を確定する必要がある。
> - Circuit Breaker ライブラリの選定（例: `pybreaker`・`tenacity`）
> - 閾値定義: 連続失敗 N 回でオープン状態遷移、クールダウン時間
> - Circuit Breaker 適用対象: netkeiba.com HTTP クライアント・Redis・PostgreSQL 接続

---

## 6. 監視・アラート

### 6-1. スクレイピング成功率監視（N-6、R-2）

| 項目 | 目標値 | アラート条件 |
|---|---|---|
| スクレイピング成功率 | ≥ 99% / 月 | 週次で閾値以下になった場合に通知（R-2 対策） |
| DB 反映遅延 | ≤ 10 分 | 超過時アラート（N-7） |
| オッズスナップショット欠損率（発走前 5 分以内） | ≤ 1% | 超過時アラート（N-8） |

### 6-2. 実行ログ管理（F-5）

```sql
-- scrape_runs テーブル（スクレイプ実行ログ）
-- カラム: target_type, status, retry_count
-- 監視基盤はこのテーブルを参照して成功率・失敗率を集計すること
```

### 6-3. API パフォーマンス監視（N-1、N-2）

| 項目 | SLO |
|---|---|
| キャッシュヒット時レスポンスタイム | ≤ 200 ms |
| キャッシュミス時レスポンスタイム | ≤ 2,000 ms |

### 6-4. 推論バッチ完了監視（N-9）

- 発走 3 時間前までに推論バッチが完了していない場合はアラートを発報する。

### 6-5. モデル品質モニタリング（Phase 4 以降）

- 特徴量重要度のモニタリング
- データドリフト検知
- 障害・エラー通知アラートの整備

### 6-6. テンポラルリーク検知（N-10）

- CI パイプラインにおいてテストデータ時系列分割によるリーク検知テストを自動実行する。
- `as_of_race_id` 紐付けの単体テストを必須化する（F-3 実装時）。

---

## 7. デプロイ

### 7-1. スキーママイグレーション（N-11）

- DDL 変更は **Alembic** でバージョン管理し、全スキーマ変更をマイグレーションファイルとして記録する。
- デプロイ時はマイグレーションプロセスを API サーバー起動前に独立実行する。

### 7-2. デプロイ順序（依存関係）

```
1. DDL マイグレーション実行（Alembic）
2. スクレイパープロセス起動
3. オッズ収集スケジューラ起動
4. 推論バッチプロセス起動
5. API サーバー起動
```

### 7-3. フェーズ別リリース計画

| Phase | 主要デプロイ対象 | 完了条件 |
|---|---|---|
| Phase 0 | scrape_runs テーブル・基本スキーマ | ラップデータ可用性確認完了 |
| Phase 1 | Layer 1〜5 全テーブル DDL・スクレイパー群・集計バッチ | 過去 2 年分データ格納済み |
| Phase 2 | 特徴量パイプライン・Stage 1 モデル・回収率計算ロジック | 勝率 Log Loss ベースライン比 −5% 改善 |
| Phase 3 | Stage 2 モデル・REST API・Redis キャッシュ・UI | 全予測ターゲット T-1〜T-11 が API 経由で取得可能 |
| Phase 4 | LSTM ラップモデル・自動再学習スケジューラ | 継続運用 |

### 7-4. キャッシュ設定（F-12、N-12）

```
キャッシュキー: prediction:{race_id}:{model_version}
キャッシュキー: lap:prediction:{race_id}:{model_version}
TTL: 発走時刻まで有効 / 発走後 60 秒で自動失効
```

---

## 8. ロールバック

### 8-1. モデルロールバック（F-16）

- 学習済みモデルはバージョン管理基盤（MLflow 等）で管理し、古いバージョンへのロールバック機能を提供する（Phase 2 で基盤整備）。
- `prediction_results` テーブルの `model_version` カラムにより、どのモデルバージョンによる推論結果かを追跡可能とする。

### 8-2. スキーマロールバック

- Alembic のダウングレード機能を用いてスキーマ変更を巻き戻す。
- Layer 2〜5 テーブルは **追記型・不変（削除不可）** 設計のため、データ自体のロールバックは行わない。

### 8-3. データロールバック非対応の設計原則

| テーブル種別 | 更新ポリシー | ロールバック可否 |
|---|---|---|
| `race_results`（Layer 2） | 追記のみ | ✗ 不変 |
| `*_stats_snapshot`（Layer 3） | 追記のみ（UNIQUE 制約） | ✗ 不変 |
| `race_odds_snapshot`（Layer 5） | 追記のみ・削除不可 | ✗ 不変 |
| `prediction_results` | UNIQUE(race_id, horse_id, model_version) | ✓ 旧 model_version を参照切替 |

---

## 9. 未決定事項（後続 DEC で確定が必要な項目）

| # | 項目 | 理由 |
|---|---|---|
| OP-1 | VPS メモリバジェット（各プロセスへの割り当て上限） | DEC-001 に記述なし |
| OP-2 | Circuit Breaker ライブラリ選定・閾値定義 | DEC-001 はリトライ設定のみ定義、CB パターン未採用 |
| OP-3 | 監視基盤ツール選定（Prometheus / Grafana / Sentry 等） | DEC-001 に記述なし |
| OP-4 | アラート通知チャネル（Slack / PagerDuty 等） | DEC-001 に記述なし |
| OP-5 | デプロイ自動化手段（GitHub Actions / Ansible 等） | DEC-001 に記述なし |
| OP-6 | MLflow 以外のモデルレジストリ候補の評価 | DEC-001 は「MLflow 等」と記載のみ |