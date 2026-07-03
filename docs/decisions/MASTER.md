# keiba-vpn — マスター仕様書
> 最終更新: 2026-07-03 | 参照: AREA-01〜AREA-09

---

## 1. プロジェクト概要

### 1-1. システム概要

netkeiba.com から収集した競馬データを用いて、出走馬ごとの **勝率・連対率・複勝率・オッズ予測・単複回収率・ポジション予測・脚質予測**、ならびに **逃げ馬ペース予測・1F 単位ラップ予測** を実現する競馬予測 Web アプリ。加えてユーザーが予想の根拠を自ら探索・検証するための **データ分析機能**（種牡馬成績分析・コース統計ダッシュボード・騎手/調教師成績分析・マイ分析）を提供する。

| 項目 | 内容 |
|---|---|
| 対象競馬 | JRA（日本中央競馬会） |
| データソース | netkeiba.com |
| ユーザー種別 | ゲスト（TOP3 閲覧のみ） / ログイン済（全頭閲覧・マイ分析保存） |
| 主要制約 | ConoHa VPS 2GB — 2GB 以内での安定稼働を最優先 |

### 1-2. 最重要原則

**`as_of_race_id` によるスナップショット管理でテンポラルリークを構造的に排除すること** が、データ基盤・モデリング・評価・API 設計の全工程の前提条件である。

### 1-3. 予測ターゲット定義

| ID | ターゲット | 問題設定 | 出力型 | Stage |
|---|---|---|---|---|
| T-1 | `win_prob`（勝率） | 多クラス分類（1着） | `NUMERIC(5,4)` | Stage 1 |
| T-2 | `place_prob`（連対率） | バイナリ分類（2着以内） | `NUMERIC(5,4)` | Stage 1 |
| T-3 | `show_prob`（複勝率） | バイナリ分類（3着以内） | `NUMERIC(5,4)` | Stage 1 |
| T-4 | `predicted_win_odds` | 回帰 | `NUMERIC(7,1)` | Stage 1 |
| T-5 | `predicted_place_odds` | 回帰 | `NUMERIC(7,1)` | Stage 1 |
| T-6 | `win_roi`（単回収率） | ポスト計算値: `win_prob × predicted_win_odds × 100` | `NUMERIC(7,2)` | ポスト処理 |
| T-7 | `show_roi`（複回収率） | ポスト計算値: `show_prob × predicted_place_odds × 100` | `NUMERIC(7,2)` | ポスト処理 |
| T-8 | `predicted_position` | 順位回帰 / ランキング学習 | `SMALLINT` | Stage 1 |
| T-9 | `predicted_running_style` | 4値分類: `FRONT`/`STALKER`/`MID`/`CLOSER` | `VARCHAR(10)` | Stage 1 |
| T-10 | `pace_category` | 3値分類: `HIGH`/`MIDDLE`/`SLOW` | `VARCHAR(10)` | Stage 2 |
| T-11 | `predicted_lap_sec[]` | 時系列回帰（1F 単位系列出力） | `NUMERIC(4,2)[]` | Stage 2 |

> T-6・T-7 はモデルの直接予測ターゲットではなく、T-1〜T-5 の推論結果に基づくポスト計算値。100 以上 = 期待値プラスのバリューベット候補。

---

## 2. 技術スタック

| レイヤー | 技術 | 備考 |
|---|---|---|
| データベース | PostgreSQL | `BIGSERIAL`・`TIMESTAMPTZ`・`NUMERIC`・`ARRAY` 型使用 |
| キャッシュ | Redis | TTL 管理付き、発走後 60 秒で自動失効 |
| スキーママイグレーション | Alembic | 全 DDL 変更をバージョン管理 |
| API フレームワーク | Flask | ベースパス `/api/v1`、レスポンス `application/json` |
| ML フレームワーク（初期） | LightGBM | Stage 1/2 共通、表形式データ向け |
| ML フレームワーク（拡張） | LSTM | Phase 4 以降、ラップ系列予測に適用 |
| モデル管理 | MLflow（または同等） | バージョニング・ロールバック対応 |
| 特徴量重要度 | TreeSHAP（`shap.TreeExplainer`） | 全 LightGBM モデルに適用 |
| テスト | pytest / pytest-cov | カバレッジ目標 ≥ 80% |
| GCS ストレージ | `gs://keiba-vpn-data/` | raw / processed / features / models / predictions / logs |
| フロントエンド | 未確定（Next.js 採用候補） | Phase 3 で確定・実装 |
| CI/CD | 未確定 | GitHub Actions 等、後続 DEC で確定 |

---

## 3. アーキテクチャ設計

### 3-1. システムコンポーネント構成

```
scraper        ── netkeiba.com スクレイパー（concurrent_workers: 1）
db (PostgreSQL) ── Layer 1〜5 データ格納
redis          ── 予測結果・ラップ予測・オッズキャッシュ
api (Flask)    ── REST API（/api/v1/...）
ml-worker      ── 特徴量生成・モデル学習・推論バッチ
mlflow         ── モデルバージョン管理
frontend       ── レース一覧・予測表示 UI（Phase 3）
```

### 3-2. データ層アーキテクチャ（5層構造）

| 層 | 名称 | 主テーブル | 更新ポリシー |
|---|---|---|---|
| Layer 1 | 静的マスター | `races`, `entries`, `horses`, `jockeys`, `trainers`, `courses`, `sires` | 追記・参照更新 |
| Layer 2 | 確定結果 | `race_results` | 追記のみ |
| Layer 3 | 集計スナップショット | `horse_stats_snapshot`, `jockey_stats_snapshot`, `trainer_stats_snapshot` | 追記のみ・`UNIQUE(entity_id, as_of_race_id)` |
| Layer 4 | ラップ・ペース・コーナー | `race_lap_times`, `race_corner_positions`, `race_pace_summary` | 追記のみ |
| Layer 5 | オッズ時系列 | `race_odds_snapshot` | 追記のみ・削除不可 |

**特徴量リーク防止の原則**: Layer 3 の集計値は必ず `as_of_race_id`（予測対象レース）に紐付けて保存し、そのレース以後の情報を含めない。

### 3-3. 2ステージ ML アーキテクチャ

```
Stage 1: 共有表現マルチタスクモデル
  入力: Layer 1〜3 特徴量（馬×レース単位）
  [Shared Encoder (LightGBM)]
    ├── Head A: 勝率/連対率/複勝率（分類）       → T-1/T-2/T-3
    ├── Head B: ポジション予測（LambdaMART）     → T-8
    └── Head C: オッズ予測（回帰）               → T-4/T-5
                    │ T-8 ポジション予測値を受け渡し
Stage 2: ラップ・ペース予測モデル
  入力: Layer 4 + Stage 1 ポジション予測 + コース形状特徴量
  出力: ペースカテゴリ（T-10）+ 1F 毎ラップ予測（T-11）
```

### 3-4. GCS バケット構成

```
gs://keiba-vpn-data/
├── raw/          # スクレイピング生データ（Layer 1〜5 原形）
├── processed/    # クリーニング・変換済みデータ（Parquet）
├── features/     # Feature Store（Layer 3 スナップショット）
├── models/       # 学習済みモデル成果物
├── predictions/  # 推論結果
└── logs/         # スクレイピング実行ログ
```

`gcs_paths.py` をパス定義の SSoT として全モジュールからインポートして使用する。パス関数の主要シグネチャ:

| 関数 | 用途 |
|---|---|
| `raw_race_card(race_id)` | 出馬表 HTML 生データ |
| `raw_race_result(race_id)` | レース結果 HTML 生データ |
| `raw_odds_snapshot(race_id, snapshot_at_iso)` | オッズスナップショット HTML |
| `feature_horse_snapshot(horse_id, as_of_race_id)` | 馬スナップショット特徴量 Parquet |
| `feature_race_combined(race_id)` | レース単位結合特徴量 Parquet |
| `model_artifact(model_name, version)` | 学習済みモデル pkl |
| `prediction_result(race_id, model_version)` | 推論結果 Parquet |

### 3-5. ETL パイプライン

```
netkeiba.com
    │ HTTP スクレイピング
    ▼
[Extract]  生HTML → GCS raw/（gcs_paths.py 経由）
    ▼
[Transform] HTML解析 → Parquet → GCS processed/
    ▼
[Load]     Parquet → PostgreSQL（Layer 1〜5）
    ▼
[Snapshot Batch] Layer 3 スナップショット生成（as_of_race_id 付与）
    ▼
GCS features/ & PostgreSQL *_stats_snapshot
```

### 3-6. 4 層キャッシュ構成

| 層 | 種別 | 対象 | TTL | キャッシュキー |
|---|---|---|---|---|
| L1 | Flask 内メモリ（`lru_cache`） | コース・マスターデータ等静的情報 | プロセス再起動まで | N/A |
| L2 | Redis — 予測結果 | T-1〜T-9 全ターゲット | 発走時刻まで / 発走後 60 秒失効 | `prediction:{race_id}:{model_version}` |
| L3 | Redis — ラップ予測 | T-10〜T-11 | 発走時刻まで / 発走後 60 秒失効 | `lap:prediction:{race_id}:{model_version}` |
| L4 | Redis — オッズスナップショット | 直近オッズ（推論特徴量用） | 5 分 | `odds:latest:{race_id}` |

### 3-7. プロセス分離

| プロセス種別 | 役割 | 備考 |
|---|---|---|
| スクレイパー | netkeiba.com からのデータ収集（Layer 1〜5） | `concurrent_workers: 1`、シングルIP 制約 |
| スナップショット集計バッチ | `*_stats_snapshot` 生成 | results 収集完了後に起動 |
| オッズ収集スケジューラ | 発走当日 08:00〜発走時刻まで収集 | タイムウィンドウ制御 |
| 推論バッチプロセス | Stage 1 → Stage 2 順次推論・結果書込 | 発走 3 時間前までに完了（N-9） |
| API サーバー（Flask） | REST API 提供・Redis キャッシュ参照 | キャッシュヒット ≤ 200 ms |
| DDL マイグレーション | Alembic スキーマバージョン管理 | デプロイ時に独立実行 |

### 3-8. デプロイ順序

```
1. DDL マイグレーション実行（Alembic）
2. スクレイパープロセス起動
3. オッズ収集スケジューラ起動
4. 推論バッチプロセス起動
5. API サーバー起動
```

---

## 4. 機能要件（確定版）

### 4-1. 予測機能

| # | 要件 | 優先度 |
|---|---|---|
| F-1 | netkeiba.com からレース基本情報・出馬表をスクレイピングして Layer 1 に格納する | 高 |
| F-2 | レース結果・ラップタイム・コーナー通過順位をスクレイピングして Layer 2/4 に格納する | 高 |
| F-3 | 馬・騎手・調教師の集計統計を `as_of_race_id` 付きスナップショットとして Layer 3 に格納する | 高 |
| F-4 | オッズを指定スケジュール（発走前 5 分毎〜1 分毎）でスナップショット取得し Layer 5 に格納する | 高 |
| F-5 | スクレイプ実行ログを `scrape_runs` テーブルで管理する | 高 |
| F-6 | 特徴量パイプラインで脚質スコア・クロス特徴量・相対特徴量を自動生成する | 高 |
| F-7 | Stage 1 モデル（勝率/連対率/複勝率/ポジション/オッズ予測）を学習・推論する | 高 |
| F-8 | Stage 2 モデル（ペースカテゴリ/1F 毎ラップ予測）を Stage 1 出力を受けて学習・推論する | 高 |
| F-9 | 推論後に単回収率・複回収率を計算し `prediction_results` に保存する | 高 |
| F-10 | 任意レースの予測結果（T-1〜T-11 全ターゲット）を REST API で提供する | 高 |
| F-11 | ラップ予測結果を系列形式（`furlong_index` 順）で API から提供する | 中 |
| F-12 | 予測結果を Redis にキャッシュし、同一リクエストの DB 再クエリを回避する | 中 |
| F-13 | レース一覧・出馬表・AI 予測を統合表示する UI を提供する | 中 |
| F-14 | 回収率 100 以上の馬をバリューベット候補としてハイライト表示する | 中 |
| F-15 | ラップ予測をグラフ（折れ線）で可視化する | 低 |
| F-16 | 学習済みモデルのバージョン管理と古いモデルへのロールバック機能 | 中 |
| F-17 | ラップデータ可用性の事前検証（サンプル 10 レースで手動確認）— Phase 0 前提条件 | 高 |

### 4-2. データ分析機能

| # | 機能名 | 優先度 |
|---|---|---|
| AN-01 | 種牡馬成績多軸フィルタリング分析（`sires` テーブル連携） | 高 |
| AN-02 | コース別・条件別統計ダッシュボード（`course_stats_cache` 利用） | 高 |
| AN-03 | 騎手/調教師成績分析 | 高 |
| AN-04 | マイ分析（フィルター条件保存・再実行、ログインユーザー限定） | 中 |

> **UI 注記要件**: 分析画面に「※ この統計はリアルタイム集計です。AI モデルが予測に使用した時点の特徴量とは異なる場合があります。」を表示すること。

### 4-3. REST API エンドポイント

| メソッド | エンドポイント | 説明 | 機能要件 |
|---|---|---|---|
| `GET` | `/api/v1/races` | レース一覧取得 | F-13 |
| `GET` | `/api/v1/races/{race_id}` | レース詳細・出馬表取得 | F-13 |
| `GET` | `/api/v1/races/{race_id}/predictions` | 全予測ターゲット（T-1〜T-9）取得 | F-10 |
| `GET` | `/api/v1/races/{race_id}/predictions/laps` | ラップ予測系列（T-10〜T-11）取得 | F-11 |

#### `GET /api/v1/races/{race_id}/predictions` レスポンス仕様

```json
{
  "race_id": "202506010811",
  "model_version": "v1.2.0",
  "predicted_at": "2025-06-01T08:30:00+09:00",
  "pace_prediction": {
    "pace_category": "MIDDLE",
    "lap_times": [
      { "furlong_index": 1, "predicted_lap_sec": 12.3 },
      { "furlong_index": 2, "predicted_lap_sec": 11.8 }
    ]
  },
  "horses": [
    {
      "horse_id": "2019105678",
      "post_no": 3,
      "win_prob": 0.1823,
      "place_prob": 0.3241,
      "show_prob": 0.4815,
      "predicted_win_odds": 5.2,
      "predicted_place_odds": 2.1,
      "expected_win_roi": 94.8,
      "expected_show_roi": 101.1,
      "predicted_position": 2,
      "predicted_running_style": "STALKER",
      "is_value_bet": true
    }
  ]
}
```

- `is_value_bet`: `expected_win_roi >= 100` または `expected_show_roi >= 100` の場合に `true`

### 4-4. スクレイピング収集スケジュール

| ジョブ名 | トリガー | 間隔 | 優先ウィンドウ |
|---|---|---|---|
| `race_card` | レース 3 日前 06:00 JST | 毎日 06:00（発走まで） | — |
| `odds_snapshot` | 発走当日 08:00〜発走時刻 | 5 分毎 | 発走 30 分前: 2 分毎 / 発走 5 分前: 1 分毎 |
| `race_results` | 発走予定時刻 + 35 分 | リトライ: 5 分間隔 × 最大 6 回 | — |
| `horse_history` | results 収集完了後 | 結果確定後 1 回 | 前走成績更新後に再取得 |

### 4-5. スクレイピング設定

```python
SCRAPING_CONFIG = {
    "request_interval_sec": 2.0,
    "jitter_sec": (0.5, 1.5),
    "concurrent_workers": 1,
    "session_rotate_interval": 50,
    "retry_on_429": True,
    "retry_backoff_base_sec": 30,
    "user_agent_rotate": True,
}
```

### 4-6. フェーズ別ロードマップ

| Phase | 主要対象 | 完了条件 |
|---|---|---|
| Phase 0 | `scrape_runs` テーブル・基本スキーマ | ラップデータ可用性確認完了（F-17） |
| Phase 1 | Layer 1〜5 全テーブル DDL・スクレイパー群・集計バッチ | 過去 2 年分データ格納済み |
| Phase 2 | 特徴量パイプライン・Stage 1 モデル・回収率計算ロジック | 勝率 Log Loss ベースライン比 −5% 改善 |
| Phase 3 | Stage 2 モデル・REST API・Redis キャッシュ・UI | 全予測ターゲット T-1〜T-11 が API 経由で取得可能 |
| Phase 4 | LSTM ラップモデル・自動再学習スケジューラ | 継続運用 |

---

## 5. 非機能要件（確定版）

| ID | 区分 | 要件 | 目標値 |
|---|---|---|---|
| N-1 | パフォーマンス | API レスポンスタイム（キャッシュヒット時） | ≤ 200 ms |
| N-2 | パフォーマンス | API レスポンスタイム（キャッシュミス時） | ≤ 2,000 ms |
| N-3 | ML 精度 | ラップタイム予測 MAE | ≤ 0.3 秒 |
| N-4 | ML 精度 | 勝率 Log Loss（ベースライン比改善） | −5% 以上 |
| N-5 | ML 精度 | ポジション予測 Spearman ρ | ≥ 0.55 |
| N-6 | 可用性 | スクレイピング成功率 | ≥ 99% / 月 |
| N-7 | 鮮度 | DB 反映遅延（スクレイピング完了から） | ≤ 10 分 |
| N-8 | 鮮度 | オッズスナップショット欠損率（発走前 5 分以内） | ≤ 1% |
| N-9 | タイミング | 推論バッチ完了タイミング | 発走 3 時間前まで |
| N-10 | 品質 | テンポラルリーク検知テストを CI に必須実装 | CI 合格必須 |
| N-11 | 管理 | 全スキーマ変更を Alembic でバージョン管理 | マイグレーションファイル必須 |
| N-12 | キャッシュ | 予測結果 Redis TTL（発走後自動失効） | 発走後 60 秒 |
| N-13 | テスト | 単体テストカバレッジ（スクレイパー・特徴量 PL・モデル・API） | ≥ 80% |
| N-14 | データ品質 | 障害レース・海外レースを予測対象外管理 | `is_excluded` フラグで明示 |

### 5-1. 監視・アラート

| 監視項目 | SLO / 閾値 | アラート条件 |
|---|---|---|
| スクレイピング成功率 | ≥ 99% / 月 | 週次で閾値以下 |
| DB 反映遅延 | ≤ 10 分 | 超過時 |
| オッズスナップショット欠損率 | ≤ 1% | 超過時 |
| API レスポンス（キャッシュヒット） | ≤ 200 ms | 超過時 |
| API レスポンス（キャッシュミス） | ≤ 2,000 ms | 超過時 |
| 推論バッチ完了 | 発走 3 時間前 | 未完了時 |

### 5-2. 認証・認可方針

| 項目 | 方針 |
|---|---|
| 認証方式 | API キー（`Authorization: Bearer <token>`）または セッション Cookie（UI 向け） |
| 認可スコープ | 予測 API・レース API は読み取り専用。スクレイプ実行・モデル管理は内部ネットワーク限定 |
| 管理系エンドポイント | `/api/v1/admin/*` は内部 IP（127.0.0.1 / VPN 内）のみ許可 |

### 5-3. テスト要件

#### CI ゲート（全項目ブロッキング）

| ゲート | ブロッキング条件 |
|---|---|
| テンポラルリーク検知 | `as_of_race_id` リーク検知テスト失敗 |
| Unit テストカバレッジ | < 80% |
| 勝率 Log Loss 改善 | < −5%（ベースライン比） |
| Spearman ρ | < 0.55 |
| ラップ MAE | > 0.3 秒 |
| API レスポンス（キャッシュヒット） | > 200 ms |
| API レスポンス（キャッシュミス） | > 2,000 ms |
| 時系列分割ランダムシャッフル検出 | `train_test_split(shuffle=True)` 使用を検知 |

#### テストデータ管理

| データ種別 | 用途 | 管理方法 |
|---|---|---|
| フィクスチャ（静的 JSON/CSV） | Unit テスト・テンポラルリーク検知 | `tests/fixtures/`（Git 管理） |
| サンプルスクレイプ HTML | スクレイパー Unit テスト | `tests/fixtures/html/`（静的保存、実サイト非依存） |
| ヒストリカルレースデータ（2〜3 年分） | Stage 1/2 学習・バックテスト | DB の `test` スキーマに格納 |
| 時系列分割検証データセット | ML テスト（分割正当性確認） | `tests/data/split/`（Git 管理） |
| Phase 0 検証用サンプル（10 レース） | ラップデータ可用性確認（F-17） | `tests/phase0/`（手動格納・レビュー対象） |

---

## 6. AI / ML パイプライン

### 6-1. アルゴリズム選定

| ターゲット | アルゴリズム | 選定理由 |
|---|---|---|
| 勝率 (T-1) | LightGBM softmax | 表形式データに最強、欠損耐性が高い |
| 連対率・複勝率 (T-2/3) | LightGBM binary | 同上 |
| ポジション予測 (T-8) | LambdaMART（LightGBM ranker） | 相対順位を直接最適化 |
| オッズ予測 (T-4/5) | LightGBM regression | マーケット形成ロジックとの親和性 |
| ペースカテゴリ (T-10) | LightGBM multiclass | 3 クラス・解釈性を重視 |
| 1F 毎ラップ予測 (T-11) | LightGBM per-furlong（初期）→ LSTM（Phase 4） | 解釈しやすい単独回帰から開始、系列依存大なら LSTM へ移行 |

### 6-2. 特徴量定義

#### 基本特徴量（Layer 1〜2 由来）

| 特徴量名 | 型 | 特徴量名 | 型 |
|---|---|---|---|
| `distance` | INT | `weight_carried` | FLOAT |
| `surface` | CATEGORY | `horse_weight` | INT |
| `direction` | CATEGORY | `horse_weight_diff` | INT |
| `going` | CATEGORY | `days_since_last` | INT |
| `weather` | CATEGORY | `horse_age` | INT |
| `grade` | CATEGORY | `sex` | CATEGORY |
| `horse_num` | INT | `frame_no` | INT |
| `post_no` | INT | — | — |

#### 集計特徴量（Layer 3 スナップショット由来）

| 特徴量名 | 説明 |
|---|---|
| `win_rate_all` / `place_rate_all` / `show_rate_all` | 生涯勝率・連対率・複勝率 |
| `win_rate_distance` / `win_rate_course` / `win_rate_going` | 条件別勝率（±200m 同距離帯 / 同コース / 同馬場状態） |
| `avg_last_3f` / `speed_index_avg` / `speed_index_max` | 直近 5 走平均上がり 3F・スピード指数 |
| `running_style_score` | 脚質スコア（−5=逃〜+5=追込） |
| `jockey.win_rate_all` | 騎手勝率 |
| `trainer.win_rate_all` | 調教師勝率 |

> **必須制約**: Layer 3 集計値は必ず `as_of_race_id = 予測対象レース ID` のスナップショットを参照すること。

#### クロス・相対特徴量（前処理で自動生成）

```python
df["style_x_straight"]    = df["running_style_score"] * df["final_straight_length"]
df["style_x_distance"]    = df["running_style_score"] * df["distance_category_encoded"]
df["front_runner_count"]  = df.groupby("race_id")["running_style_score"].transform(lambda x: (x < -2).sum())
df["rel_speed_index"]     = df["speed_index_avg"] / df.groupby("race_id")["speed_index_avg"].transform("mean")
df["rel_days_since_last"] = df["days_since_last"] - df.groupby("race_id")["days_since_last"].transform("mean")
df["rel_odds_rank"]       = df.groupby("race_id")["odds_value"].rank(ascending=True)
df["pace_scenario_prior"] = (df["front_runner_count"] / df["horse_num"]).apply(
    lambda r: "HIGH" if r > 0.3 else ("SLOW" if r < 0.1 else "MIDDLE"))
```

### 6-3. 学習パイプライン

- 時系列順に train / validation / test を分割。**ランダムシャッフル禁止**（CI で静的チェック）。
- 常に過去レースで学習し、未来レースで評価する。
- 推論時は `as_of_race_id = 対象レース ID` のスナップショットのみ使用する。
- オッズ特徴量は「発走 N 分前の最終スナップショット」を固定使用（直前確定値）。

### 6-4. 回収率ポスト計算ロジック

```python
def calculate_recovery_rate(
    win_prob: float, win_odds: float,
    show_prob: float, place_odds_mid: float
) -> dict:
    win_roi  = win_prob  * win_odds       * 100   # T-6
    show_roi = show_prob * place_odds_mid * 100   # T-7
    return {"win_roi": round(win_roi, 2), "show_roi": round(show_roi, 2)}
```

### 6-5. ModelRegistry・バージョニング

- **MLflow**（または同等ツール）でモデル管理（F-16）。
- `prediction_results` テーブルの `model_version` カラムで推論結果を追跡。
- SHAP 値（TreeSHAP）をモデルバージョンに紐付けてアーティファクトとして保存。
- 旧バージョンへのロールバックは `model_version` 参照切替で対応。

### 6-6. MLパイプラインと分析バッチの時点整合性分離

| 用途 | `as_of` 制約 | 集計範囲 |
|---|---|---|
| AI 予測モデル特徴量 | **必須**（`race_date < as_of_race_id`） | 予測時点以前のみ |
| ユーザー向け分析 UI | **不要** | 全期間またはUI選択 |

### 6-7. 評価指標

| ターゲット | 主要指標 | 補助指標 |
|---|---|---|
| 勝率 (T-1) | Log Loss（ベースライン比 −5%） | Calibration Error, Top-1 Accuracy |
| 連対率・複勝率 (T-2/3) | Binary Log Loss | AUC-ROC, Calibration |
| ポジション (T-8) | Spearman ρ ≥ 0.55 | MAE |
| オッズ予測 (T-4/5) | MAE（オッズ単位） | RMSE |
| ラップタイム (T-11) | MAE ≤ 0.3 秒 | RMSE per furlong |
| ペースカテゴリ (T-10) | Accuracy | Macro F1 |
| 回収率バックテスト | 通算 ROI（プラス） | Sharpe Ratio 記録 |

---

## 7. 運用コスト

### 7-1. 使用コンポーネント（確定済み）

| コンポーネント | 用途 | 推論方式 |
|---|---|---|
| PostgreSQL | Layer 1〜5 データ格納 | オンプレミス |
| Redis | 予測結果キャッシュ（TTL 発走後 60 秒失効） | オンプレミス |
| LightGBM | Stage 1/2 モデル推論 | セルフホスト |
| LSTM（Phase 4） | ラップ系列予測 | セルフホスト |
| Flask API | 予測結果配信（≤ 200 ms / ≤ 2,000 ms） | オンプレミス |
| MLflow | モデルバージョン管理 | セルフホスト |
| スクレイピングワーカー | netkeiba.com 定期取得（並列 1、間隔 2 秒） | オンプレミス |

> AI 推論は外部 API（OpenAI 等）に委託しない。すべて LightGBM / LSTM によるセルフホスト推論として設計する。

### 7-2. 未確定コスト項目（Human 判断待ち）

| 項目 | 状態 |
|---|---|
| 月額費用内訳（サーバー・DB・Redis・CDN 等） | **未定義** |
| VPS スペック・各プロセスへのメモリ割り当て上限 | **未定義** |
| スケールアップ判断基準（CPU・メモリ・レイテンシ閾値） | **未定義** |
| AI 推論外部化コスト比較（セルフホスト vs. SageMaker / Vertex AI 等） | **未定義** |
| コスト削減方針（スポットインスタンス・コールドストレージ移行等） | **未定義** |

---

## 8. 未解決事項・Human 判断待ち

| ID | カテゴリ | 項目 | 補足 |
|---|---|---|---|
| H-01 | インフラ | VPS メモリバジェット（各プロセスへの割り当て上限） | ConoHa VPS 2GB 制約内での LightGBM / Redis / PostgreSQL 割り当てを確定する必要がある |
| H-02 | インフラ | Circuit Breaker ライブラリ選定・閾値定義 | 候補: `pybreaker`・`tenacity`。連続失敗 N 回でオープン遷移、クールダウン時間を決定する必要がある |
| H-03 | 監視 | 監視基盤ツール選定 | 候補: Prometheus / Grafana / Sentry 等 |
| H-04 | 監視 | アラート通知チャネル | 候補: Slack / PagerDuty 等 |
| H-05 | CI/CD | デプロイ自動化手段 | 候補: GitHub Actions / Ansible 等 |
| H-06 | ML | MLflow 以外のモデルレジストリ候補の評価 | DEC-001 は「MLflow 等」と記載のみ |
| H-07 | コスト | 月額費用内訳・スケールアップ判断基準 | 新規 DEC で確定が必要 |
| H-08 | フロントエンド | フレームワーク確定（Next.js バージョン・Router 方式） | Phase 3 開始前に後続 DEC で決定 |
| H-09 | フロントエンド | レンダリング戦略（ISR / CSR / SSG / SSR と revalidate 間隔） | 同上 |
| H-10 | フロントエンド | PWA 対応・Lighthouse スコア目標・Core Web Vitals 目標値 | 同上 |
| H-11 | フロントエンド | グラフライブラリ選定（F-15 折れ線グラフ実装） | 候補: Recharts / Chart.js / Victory 等 |
| H-12 | フロントエンド | デザインシステム・コンポーネントライブラリ | 同上 |
| H-13 | 開発環境 | dev / stg / prod 環境分離方針 | docker-compose 設計・環境変数管理方法を含む |
| H-14 | 開発環境 | GPU 環境要件（CUDA バージョン、GPU メモリ）— Phase 4 LSTM 対応前に確定が必要 | — |

---

## 9. 参照 AREA 一覧

| AREA ID | タイトル | 最終更新 | ステータス |
|---|---|---|---|
| AREA-01 | アプリケーション要件（予測ターゲット・データ要件・機能要件・スキーマ定義・分析機能） | 2026-07-03 | FINAL |
| AREA-02 | フロントエンド要件（Next.js, ISR/CSR/SSG, UX設計, PWA, Lighthouse, パフォーマンス最適化） | 2026-07-03 | FINAL |
| AREA-03 | バックエンド要件（Flask API, DB スキーマ, 認証・認可, 4 層キャッシュ設計, レート制限） | 2026-07-03 | FINAL |
| AREA-04 | 運用最適化要件（プロセス分離 / VPS メモリバジェット / Circuit Breaker / 監視・アラート / デプロイ / ロールバック） | 2026-07-03 | FINAL |
| AREA-05 | コスト計算要件（月額費用・スケールアップ判断・AI 推論外部化比較・コスト削減方針） | 2026-07-03 | FINAL |
| AREA-06 | データ管理要件（GCS パス設計 gcs_paths.py SSoT, ETL パイプライン, Feature Store, Redis TTL 設計） | 2026-07-03 | FINAL |
| AREA-07 | モデリング管理要件（LightGBM バッチ推論, 学習パイプライン, SHAP, ModelRegistry, バージョニング, CI ゲート） | 2026-07-03 | FINAL |
| AREA-08 | テスト要件（Unit/Integration/E2E/ML テスト, CI ゲート, カバレッジ目標, テストデータ管理） | 2026-07-03 | FINAL |
| AREA-09 | 開発環境要件（実行環境前提条件・コンポーネント構成・未定義事項） | 2026-07-03 | FINAL |