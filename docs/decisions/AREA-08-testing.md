# AREA-08 — テスト要件（Unit/Integration/E2E/MLテスト, CI ゲート, カバレッジ目標, テストデータ管理）
**Status**: ACTIVE | **Last Updated**: 2026-07-06 | **Consolidates**: DEC-001（統合済み）

---

## 1. 概要

本仕様書は keiba-vpn プロジェクトにおけるテスト戦略全体を定義する。  
DEC-001 が定める「`as_of_race_id` によるスナップショット管理でテンポラルリークを構造的に排除すること」を最重要テスト観点とし、すべてのテストレベルでこの前提条件の遵守を検証する。

---

## 2. テストレベル定義

### 2-1. Unit テスト

対象コンポーネントと検証事項を以下に示す。

| 対象モジュール | 主要テストケース |
|---|---|
| `horse_stats_snapshot` 集計バッチ | `as_of_race_id` に対して対象レース以後の情報が含まれないこと（テンポラルリーク検知） |
| `jockey_stats_snapshot` / `trainer_stats_snapshot` 集計バッチ | 同上 |
| `calculate_recovery_rate()` | `win_roi = win_prob × win_odds × 100`、`show_roi = show_prob × place_odds_mid × 100` の計算精度（小数点2桁） |
| 特徴量パイプライン | `running_style_score`（−5〜+5 の範囲）、`front_runner_count`、`pace_scenario_prior`、`rel_speed_index`、`rel_odds_rank` の生成値が仕様通りであること |
| `race_pace_summary` 生成ロジック | `pace_category` が `HIGH` / `MIDDLE` / `SLOW` のいずれかであること |
| スクレイパー HTML パーサー | 各 URL パターン（`/race/shutuba/{race_id}/`、`/race/{race_id}/`、`/horse/{horse_id}/`、`/jockey/{jockey_id}/`、`/trainer/{trainer_id}/`、`/odds/{race_id}/`）に対してフィールド抽出が正しいこと |
| `scrape_runs` 実行管理 | `target_type`・`status`・`retry_count` の記録が正確であること |
| Redis キャッシュキー生成 | `prediction:{race_id}:{model_version}` / `lap:prediction:{race_id}:{model_version}` の形式が正しいこと |

### 2-2. Integration テスト

| テスト対象 | 検証内容 |
|---|---|
| スクレイパー → Layer 1〜5 DB 格納フロー | スクレイプ後 10 分以内に対象テーブルへ反映されること（N-7: ≤ 10 分） |
| Layer 3 スナップショット → Stage 1 モデル 特徴量ロード | `as_of_race_id = 対象レースID` のスナップショットのみが特徴量として読み込まれること |
| Stage 1 出力（ポジション予測）→ Stage 2 入力連携 | `predicted_position` が Stage 2 の入力特徴量として正しく受け渡されること |
| 推論パイプライン → `prediction_results` 保存 | T-1〜T-9 全フィールドが `prediction_results` テーブルに `UNIQUE(race_id, horse_id, model_version)` 制約を満たして保存されること |
| 推論パイプライン → `prediction_lap_times` 保存 | `furlong_index` 順に `predicted_lap_sec` / `predicted_pace_cat` が保存されること |
| オッズスナップショット → 推論時特徴量選択 | 「発走 N 分前の最終スナップショット」が一意に選択されること |
| Redis キャッシュ TTL | 発走前はキャッシュが有効であり、発走後 60 秒で自動失効すること |
| Alembic マイグレーション | 全 DDL マイグレーションが冪等に適用・ロールバックできること |

### 2-3. E2E テスト

| シナリオ | 期待結果 |
|---|---|
| `GET /api/v1/races/{race_id}/predictions` （キャッシュヒット） | レスポンスタイム ≤ 500 ms (p95)、T-1〜T-9 全フィールドを含む JSON を返すこと（N-1） |
| `GET /api/v1/races/{race_id}/predictions` （キャッシュミス） | レスポンスタイム ≤ 2,000 ms（N-2） |
| ラップ予測エンドポイント | `pace_prediction.lap_times` に `furlong_index` 順の系列データが含まれること |
| バリューベット候補フラグ | `expected_win_roi ≥ 100` または `expected_show_roi ≥ 100` の馬に `is_value_bet: true` が付与されること |
| 障害レース除外 | 障害レースの `race_id` に対して予測対象外フラグが返され、推論が実行されないこと（N-14） |
| 発走後 API レスポンス | 発走後 60 秒経過後は Redis キャッシュがなく DB から新鮮なデータが返ること |
| F-6 レスポンシブテスト（Playwright） | モバイル・デスクトップ両ビューポートで主要画面（出馬表・予測結果・分析ダッシュボード）が崩れず表示されること（Playwright ブラウザ E2E） |

### 2-4. ML テスト

ML テストは「テンポラルリーク防止」を最優先とし、以下を実施する。

#### 2-4-1. テンポラルリーク検知テスト（CI 必須）

```python
def test_no_temporal_leak_in_snapshot():
    """
    horse_stats_snapshot の各レコードについて、
    as_of_race_id のレース開催日より新しい race_result が
    集計に含まれていないことを検証する。
    """
    for snapshot in fetch_all_snapshots():
        race_date = get_race_date(snapshot.as_of_race_id)
        assert snapshot.as_of_date <= race_date, (
            f"Temporal leak detected: snapshot for horse {snapshot.horse_id} "
            f"as_of {snapshot.as_of_race_id} contains future data"
        )
```

#### 2-4-2. 時系列分割テスト

- 学習データの最終レース開催日 < 検証データの最初のレース開催日 < テストデータの最初のレース開催日 であることをアサート
- ランダムシャッフルによる分割がコードパスに存在しないことを静的チェック（`train_test_split(shuffle=True)` の使用禁止）

#### 2-4-3. モデル精度ゲート

CI パイプラインで以下の閾値を下回った場合はデプロイをブロックする。

| ターゲット | 指標 | 合格閾値 |
|---|---|---|
| 勝率 (T-1) | Log Loss（ベースライン比） | ベースライン比 −5% 以上改善（N-4） |
| ポジション予測 (T-8) | Spearman ρ | ≥ 0.55（N-5） |
| ラップタイム (T-11) | MAE（秒） | ≤ 0.3 秒（N-3） |
| 回収率バックテスト | 通算 ROI | プラス、かつ Sharpe Ratio を記録 |

#### 2-4-4. 評価指標テスト

| ターゲット | 検証する評価指標 | 補助指標 |
|---|---|---|
| 勝率 (T-1) | Log Loss | Calibration Error, Top-1 Accuracy |
| 連対率・複勝率 (T-2/3) | Binary Log Loss | AUC-ROC, Calibration |
| ポジション (T-8) | Spearman ρ | MAE |
| オッズ予測 (T-4/5) | MAE（オッズ単位） | RMSE |
| ラップタイム (T-11) | MAE（秒） | RMSE per furlong |
| ペースカテゴリ (T-10) | Accuracy | Macro F1 |

---

## 3. CI ゲート

### 3-1. ゲート構成

```yaml
# CI パイプライン ゲート定義（概要）
gates:
  - name: lint_and_static_analysis
    blocking: true
    checks:
      - no_shuffle_in_time_split   # train_test_split(shuffle=True) 禁止
      - schema_migration_valid     # Alembic マイグレーション構文チェック

  - name: unit_tests
    blocking: true
    coverage_threshold: 80        # カバレッジ目標（後述）
    targets:
      - scraper/
      - feature_pipeline/
      - models/
      - api/

  - name: temporal_leak_detection
    blocking: true                # テンポラルリーク検知は必須ゲート
    targets:
      - tests/ml/test_temporal_leak.py

  - name: integration_tests
    blocking: true
    targets:
      - tests/integration/

  - name: ml_quality_gates
    blocking: true
    checks:
      - logloss_improvement_vs_baseline  # ≥ −5%
      - spearman_rho                     # ≥ 0.55
      - lap_mae                          # ≤ 0.3 秒

  - name: e2e_tests
    blocking: true
    targets:
      - tests/e2e/
    performance_checks:
      - api_latency_cache_hit_p95_ms: 500
      - api_latency_cache_miss_ms: 2000

  - name: responsive_tests        # F-6
    blocking: true
    tool: playwright
    viewports:
      - mobile                    # 例: 375×812 (iPhone 13)
      - desktop                   # 例: 1280×800
    targets:
      - tests/e2e/responsive/
```

### 3-2. ブロッキング条件まとめ

| ゲート | ブロッキング | 理由 |
|---|---|---|
| テンポラルリーク検知テスト失敗 | ✅ ブロック | DEC-001 の最重要事項（`as_of_race_id` リーク排除）の違反 |
| Unit テストカバレッジ < 80% | ✅ ブロック | N-13 の要件（スクレイパー・特徴量パイプライン ≥ 80%） |
| 勝率 Log Loss 改善 < −5% | ✅ ブロック | N-4 の精度要件 |
| Spearman ρ < 0.55 | ✅ ブロック | N-5 の精度要件 |
| ラップ MAE > 0.3 秒 | ✅ ブロック | N-3 の精度要件 |
| API レスポンス（キャッシュヒット）> 500 ms (p95) | ✅ ブロック | N-1 の非機能要件 |
| API レスポンス（キャッシュミス）> 2,000 ms | ✅ ブロック | N-2 の非機能要件 |
| 時系列分割ランダムシャッフル検出 | ✅ ブロック | テンポラルリーク防止ルールの遵守 |

---

## 4. カバレッジ目標

### 4-1. 全体目標

DEC-001 N-13 に基づき、以下のカバレッジ目標を設定する。

| 対象モジュール | カバレッジ目標 | 計測ツール |
|---|---|---|
| スクレイパー（`scraper/`） | **≥ 80%** | pytest-cov |
| 特徴量パイプライン（`feature_pipeline/`） | **≥ 80%** | pytest-cov |
| モデル学習・推論（`models/`） | **≥ 80%** | pytest-cov |
| API レイヤー（`api/`） | **≥ 80%** | pytest-cov |
| データ集計バッチ（`batch/`） | **≥ 80%** | pytest-cov |

### 4-2. 重点カバレッジ対象

以下のコードパスは 100% カバレッジを目指す（テンポラルリーク防止の中核ロジック）。

- `horse_stats_snapshot` / `jockey_stats_snapshot` / `trainer_stats_snapshot` の `as_of_race_id` フィルタリングロジック
- `calculate_recovery_rate()` 関数
- オッズスナップショット「発走 N 分前の最終スナップショット」選択ロジック

---

## 5. テストデータ管理

### 5-1. テストデータの種類と用途

| データ種別 | 用途 | 管理方法 |
|---|---|---|
| フィクスチャ（静的 JSON/CSV） | Unit テスト・テンポラルリーク検知テスト | `tests/fixtures/` 以下に版管理（Git） |
| サンプルスクレイプ HTML | スクレイパー Unit テスト | `tests/fixtures/html/` 以下に静的保存（実サイト非依存） |
| ヒストリカルレースデータ（2〜3 年分） | Stage 1/2 モデル学習・バックテスト | DB（本番と別の `test` スキーマ）に格納 |
| 時系列分割検証データセット | ML テスト（時系列分割正当性確認） | `tests/data/split/` 以下に版管理 |
| Phase 0 検証用サンプル（10 レース） | ラップデータ可用性確認（F-17） | `tests/phase0/` 以下に手動格納・レビュー対象 |

### 5-