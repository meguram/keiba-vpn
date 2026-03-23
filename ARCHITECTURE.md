# ML-AutoPilot Keiba — システム仕様書

> 自動生成: 2026-03-19 01:36:10
> このファイルはシステム起動時・コマンド実行時に自動更新されます。

---

## 1. 設計思想

### 1.1 階層型マルチエージェントアーキテクチャ

本システムは**将軍 → 家老 → 足軽**の3階層構造で設計されている。

| 階層 | 呼称 | 責務 |
|------|------|------|
| **将軍** (Shogun) | Orchestrator | ユーザー入力の意図分類、家老への委譲、結果の返却 |
| **家老** (Karou) | MiddleManager | 担当領域のサブエージェント連携・調整、他家老との協調 |
| **足軽** (Ashigaru) | BaseAgent | 単一責務のタスク実行 (学習, 予測, スクレイピング等) |

### 1.2 設計原則

- **委譲原則**: 将軍はロジックを持たず、全てを家老に委譲する
- **自律性**: 各家老は独立して配下のエージェントを統率できる
- **安全デフォルト**: 認識不能なコマンドは予測を自動実行しない (`unknown → 選択肢提示`)
- **疎結合通信**: エージェント間通信は MessageBus 経由、家老間連携は明示的な API 呼び出し
- **観測可能性**: 全エージェントの状態を StatusBoard と MLflow で追跡可能

## 2. 階層構成

```
将軍 (Orchestrator)
│
├── 意図分類エンジン (parse_command)
│     スコアリング方式で 6カテゴリ × 20+ アクションに分類
│
├── 軍師 (ConfigMgr)
│    責務: 設定・要件管理
│    配下:
│      └── RDA (要件定義)
│    • 蓄積要件の管理
│    • MLP仕様 (MLProjectSpec) の生成
│    • 設定表示・変更・リセット
│
├── パイプライン奉行 (PipelineMgr)
│    責務: 予測パイプライン統率
│    配下:
│      ├── SMA (データ取得)
│      ├── DOA (データ整理)
│      ├── FEA (特徴量生成)
│      ├── MTEA (モデル学習)
│      └── POA (予測出力)
│    • 軍師から仕様を受け取り全6フェーズ実行
│    • SMA → DOA → FEA → MTEA → POA のシーケンス制御
│    • 学習結果は MLflow に自動記録
│
├── データ奉行 (DataMgr)
│    責務: データ取得・確認
│    • ScraperRunner 統合 (5種類のスクレイピング)
│    • データプレビュー・統計・カラム一覧
│    • 予測結果エクスポート
│
└── 目付 (InfoMgr)
     責務: 情報照会・報告
     • 他3家老から情報を収集
     • エージェント状態・精度・重要度の報告
     • システム全体概要の生成
```

## 3. エージェントカタログ

### 3.0 将軍 (Orchestrator)

| 属性 | 値 |
|------|-----|
| クラス | `Orchestrator` |
| ファイル | `agents/orchestrator.py` |
| 責務 | 意図分類、家老への委譲、結果返却 |
| 現在状態 | 待機中 |

### 3.1 軍師 (ConfigMgr)

| 属性 | 値 |
|------|-----|
| クラス | `ConfigManager` |
| ファイル | `agents/config_manager.py` |
| 責務 | 設定・要件管理を統率する家老 |
| 入力 | ユーザーの設定変更コマンド、自然言語要件 |
| 出力 | MLProjectSpec、設定レポート |
| 公開API | `generate_spec()`, `handle_show()`, `handle_change()`, `handle_reset()`, `add_requirements()` |
| 現在状態 | 待機中 |
| 配下エージェント | RDA (要件定義) |

### 3.2 パイプライン奉行 (PipelineMgr)

| 属性 | 値 |
|------|-----|
| クラス | `PipelineManager` |
| ファイル | `agents/pipeline_manager.py` |
| 責務 | ML予測パイプラインの全6フェーズを統率する家老 |
| 入力 | MLProjectSpec (軍師経由) |
| 出力 | PredictionResult、MLflow run_id |
| 公開API | `run_pipeline(spec)` → `(PredictionResult, elapsed)` |
| 現在状態 | 待機中 |
| 配下エージェント | SMA (データ取得), DOA (データ整理), FEA (特徴量生成), MTEA (モデル学習), POA (予測出力) |

### 3.3 データ奉行 (DataMgr)

| 属性 | 値 |
|------|-----|
| クラス | `DataManager` |
| ファイル | `agents/data_manager.py` |
| 責務 | スクレイピング・データ確認・エクスポートを統率する家老 |
| 入力 | スクレイピングコマンド (action, params) |
| 出力 | スクレイピング結果、データ概要レポート |
| 公開API | `handle_scrape()`, `handle_preview()`, `handle_stats()`, `handle_columns()`, `handle_export()` |
| 現在状態 | 待機中 |

### 3.4 目付 (InfoMgr)

| 属性 | 値 |
|------|-----|
| クラス | `InfoManager` |
| ファイル | `agents/info_manager.py` |
| 責務 | 情報照会・評価報告を統率する家老 |
| 入力 | 他3家老への参照 (`set_peers()`) |
| 出力 | 状態レポート、評価レポート、概要レポート |
| 公開API | `handle_status()`, `handle_evaluate()`, `handle_importance()`, `handle_summary()` |
| 現在状態 | 待機中 |

### 3.5 足軽エージェント一覧

| ID | クラス | ファイル | 責務 | 入力 | 出力 |
|-----|--------|---------|------|------|------|
| RDA | `RequirementsDefinitionAgent` | `agents/requirements_agent.py` | 自然言語 → MLProjectSpec の構造化変換 | 自然言語テキスト, UserRequirement リスト | MLProjectSpec |
| SMA | `ScrapingMonitoringAgent` | `agents/scraping_agent.py` | netkeiba.com からのデータ取得・ファイル監視 | MLProjectSpec | ScrapingResult (CSV パス群) |
| DOA | `DataOrganizationAgent` | `agents/data_organization_agent.py` | CSV/JSON データの結合・分割・クレンジング | MLProjectSpec, ScrapingResult | OrganizedData (training_data, prediction_entries, prediction_races) |
| FEA | `FeatureEngineeringAgent` | `agents/feature_engineering_agent.py` | 特徴量生成・Label Encoding・派生特徴量・特徴量選択 | MLProjectSpec, OrganizedData | FeatureSet (X_train, X_val, y_train, y_val, X_predict, predict_meta) |
| MTEA | `ModelTrainingAgent` | `agents/model_training_agent.py` | LightGBM + Optuna チューニング・評価・MLflow 記録 | MLProjectSpec, FeatureSet | TrainedModel (model, evaluation, best_params, feature_importance, mlflow_run_id) |
| POA | `PredictionOutputAgent` | `agents/prediction_agent.py` | 学習済みモデルによる予測実行・結果構造化・JSON 出力 | TrainedModel, FeatureSet, upcoming_races | PredictionResult (races, model_evaluation, feature_importance) |


## 4. 意図分類とルーティング

### 4.1 分類フロー

```
ユーザー入力
    │
    ▼
parse_command() ── スコアリング方式
    │
    ├── 各カテゴリのキーワードマッチ数を計算
    ├── 最高スコアのカテゴリを採用
    ├── カテゴリ内で最高スコアのアクションを特定
    ├── スコア 0 かつ要件パーサーが認識 → config/change
    └── スコア 0 かつ要件なし → unknown
```

### 4.2 カテゴリ → 家老マッピング

| カテゴリ | 委譲先 | アクション |
|---------|--------|-----------|
| `pipeline` | パイプライン奉行 | predict, retrain |
| `config` | 軍師 | show, target, feature, filter, model_param, change |
| `data` | データ奉行 | preview, stats, columns |
| `scrape` | データ奉行 | race_result, race_card, horse, speed_index, race_detail, scrape |
| `info` | 目付 | status, evaluate, importance |
| `system` | 将軍/軍師/データ奉行/目付 | server, help, reset, export, summary |
| `unknown` | 将軍 | 選択肢を提示 (予測を実行しない) |

### 4.3 曖昧入力の処理

「やって」「お願い」等の曖昧な pipeline トリガーが、config/scrape 等の具体的カテゴリと
同時にマッチした場合、**具体的なカテゴリを優先する**。

```
「芝のみにしてお願い」
  → pipeline スコア 1 (「お願い」) + config スコア 1 (「芝のみ」)
  → config を優先 → 軍師に委譲
```

## 5. 予測パイプラインフロー

```
ユーザー: 「予測して」
    │
    ▼
将軍: parse_command → pipeline/predict
    │
    ▼
軍師 (ConfigManager)
    │  generate_spec()
    │  └── RDA.execute(accumulated_reqs)
    │      └── MLProjectSpec を生成
    ▼
パイプライン奉行 (PipelineManager)
    │  run_pipeline(spec)
    │
    ├── Phase 2: SMA.execute(spec)
    │   └── モックデータ or スクレイピングデータを読み込み
    │
    ├── Phase 3: DOA.execute(spec, scraping_result)
    │   └── training_data, prediction_entries, prediction_races に分割
    │
    ├── Phase 4: FEA.execute(spec, organized)
    │   └── Label Encoding → 派生特徴量 → 特徴量選択 → Train/Val/Predict 分割
    │
    ├── Phase 5: MTEA.execute(spec, feature_set)
    │   ├── Optuna チューニング (n_trials 回)
    │   ├── LightGBM 最終学習
    │   ├── 評価 (Accuracy, F1, ROC AUC)
    │   ├── モデル保存 (joblib → models/keiba_model.pkl)
    │   └── MLflow 記録 (パラメータ, メトリクス, モデル登録)
    │
    └── Phase 6: POA.execute(trained_model, feature_set, upcoming_races)
        ├── predict_proba で確率を算出
        ├── レースごとにランキング生成
        ├── 推奨印 (◎○▲△☆) を付与
        └── predictions.json に保存
```

## 6. 通信パターン

### 6.1 家老間連携

| パターン | 発信元 | 受信先 | データ | タイミング |
|---------|--------|--------|--------|-----------|
| 仕様生成 | パイプライン奉行 | 軍師 | `generate_spec()` → `MLProjectSpec` | パイプライン開始時 |
| 情報収集 | 目付 | 軍師 | `accumulated_reqs`, `last_spec` | `handle_summary()` |
| 情報収集 | 目付 | パイプライン奉行 | `last_result` | `handle_evaluate()` |
| 情報収集 | 目付 | データ奉行 | `get_data_summary()` | `handle_summary()` |
| エクスポート | データ奉行 | パイプライン奉行 | `last_result` (将軍経由) | `handle_export()` |

### 6.2 MessageBus

全エージェント (将軍・家老・足軽) が `MessageBus` を共有し、以下を発行する:

- **report**: 進捗・完了報告 (`self.report("メッセージ")`)
- **status**: 状態遷移 (`self.update_status(AgentStatus.RUNNING, "...")`)

ログは `logs/agents/` にファイル保存され、StatusBoard で Markdown ダッシュボード化される。

### 6.3 StatusBoard

`status/dashboard.md` に全エージェントのリアルタイム状態を Markdown テーブルで出力する。
パイプライン実行中は現在のフェーズ (Phase 1/6 ~ 6/6) を表示する。

## 7. データアーキテクチャ

```
data/
├── mock/                      モックデータ (CSV, 学習用)
│   ├── races.csv              レース情報 (race_id, date, racecourse, ...)
│   ├── results.csv            出走結果 (race_id, horse_name, finish_position, ...)
│   ├── upcoming_races.csv     予測対象レース
│   └── upcoming_entries.csv   予測対象出走馬
│
├── scraped/                   スクレイピング済 (JSON)
│   ├── races/                 レース結果 (db.netkeiba.com/race/{id})
│   ├── cards/                 出馬表 (race.netkeiba.com/race/shutuba.html)
│   ├── horses/               馬情報 (db.netkeiba.com/horse/{id})
│   ├── speed/                タイム指数 [要ログイン]
│   ├── past/                 馬柱 (shutuba_past.html)
│   ├── details/             出馬表+タイム指数+馬柱 統合データ
│   └── race_lists/          日別レース一覧
│
└── processed/                 加工済データ
    └── predictions.json       予測結果
```

### 7.1 データフロー

```
netkeiba.com ──[ScraperRunner]──→ GCS (JSON + HTML.gz)
                                       │
data/mock/ (CSV) ────────────┐        │
                              ▼        ▼
                         SMA (DataAgent)
                              │
                              ▼
                         DOA (整理・結合)
                              │
                    ┌─────────┼──────────┐
                    ▼         ▼          ▼
             training_data  prediction  prediction
                            _entries    _races
                    │         │
                    ▼         │
               FEA (特徴量)   │
                    │         │
                    ▼         ▼
               MTEA (学習) → POA (予測)
                    │         │
                    ▼         ▼
               MLflow     predictions.json
```

## 8. MLflow 統合

### 8.1 記録内容

| 種別 | 内容 |
|------|------|
| パラメータ | LightGBM ハイパーパラメータ, n_optuna_trials, target_variable, n_features, train_size |
| メトリクス | accuracy, f1_score, roc_auc, positive_rate |
| アーティファクト | LightGBM モデル (`model/`), 特徴量情報 (`feature_info/`) |
| タグ | agent=MTEA, target, filters |
| モデル登録 | `keiba-lgbm` (Model Registry に自動登録) |

### 8.2 モデルライフサイクル

```
MTEA 学習完了
    │
    ├── joblib.dump → models/keiba_model.pkl (ローカル)
    │
    └── mlflow.lightgbm.log_model
        ├── Tracking Server に記録
        └── Model Registry に登録 (keiba-lgbm v{N})
            │
            ├── API 予測時: load_latest_model() で最新バージョンをロード
            └── Production 昇格: MLflow UI で手動操作
```

### 8.3 フォールバック

MLflow サーバーに接続できない場合、ローカルファイルストア (`mlruns/`) に自動フォールバック。
接続テストは 3秒タイムアウトで判定。

### 8.4 MLflow ユーティリティ (`utils/mlflow_client.py`)

| 関数 | 説明 |
|------|------|
| `init_mlflow(uri?)` | Tracking URI 設定、サーバー接続確認、フォールバック |
| `get_or_create_experiment(name?)` | 実験の取得/作成、experiment_id を返却 |
| `log_training_run(params, metrics, model, ...)` | パラメータ・メトリクス・モデル登録を一括記録 |
| `load_latest_model(name?)` | Model Registry から最新モデルをロード → `(model, run_info)` |
| `get_experiment_history(limit?)` | 直近の学習履歴を取得 |

## 9. MLflow UI (ブラウザ)

MLflow Tracking Server はブラウザベースの管理 UI を提供する。

### 9.1 アクセス URL

| 環境 | URL | 認証 |
|------|-----|------|
| ローカル開発 | `http://localhost:5000` | なし |
| Docker (内部) | `http://mlflow:5000` | なし |
| 外部公開 (nginx経由) | `http://<your-ip>:80/` | Basic認証 |
| 外部公開 (HTTPS) | `https://<your-domain>/` | Basic認証 + TLS |

### 9.2 主要画面

| 画面 | パス | 機能 |
|------|------|------|
| **Experiments** | `/` | 実験一覧、run の検索・フィルタ・比較 |
| **Run 詳細** | `/#/experiments/{id}/runs/{run_id}` | パラメータ・メトリクス・アーティファクト・タグの閲覧 |
| **メトリクス比較** | 複数 run 選択 → Compare | グラフでメトリクスを比較 (Accuracy, F1, AUC の推移) |
| **Artifacts** | Run 詳細 → Artifacts タブ | モデルファイル (`model/`)、特徴量情報 (`feature_info/`) |
| **Model Registry** | `/models` | 登録モデル一覧、バージョン管理、ステージ遷移 |

### 9.3 Model Registry のステージ管理

```
None → Staging → Production → Archived
```

| ステージ | 用途 |
|---------|------|
| **None** | 学習直後に自動登録された状態 |
| **Staging** | 検証中 (手動で昇格) |
| **Production** | 本番運用モデル (API が自動で最新版をロード) |
| **Archived** | 過去バージョン |

> ステージ遷移は MLflow UI の Model Registry 画面から手動で行う。
> `load_latest_model()` はステージに関係なく最新バージョンをロードする。

### 9.4 記録されるデータの確認方法

| 確認したい内容 | 操作 |
|--------------|------|
| 学習パラメータ | Run 詳細 → Parameters タブ |
| 評価メトリクス | Run 詳細 → Metrics タブ / チャート |
| 特徴量重要度 | Run 詳細 → Artifacts → `feature_info/` |
| モデルファイル | Run 詳細 → Artifacts → `model/` |
| 複数 run 比較 | Experiments → チェック → Compare |
| モデルバージョン | Model Registry → `keiba-lgbm` → Versions |

## 10. Web ダッシュボード (ブラウザ)

### 10.1 アクセス方法

| 起動方法 | コマンド | URL |
|---------|---------|-----|
| ローカル | `python main.py --server` | `http://localhost:8000` |
| Docker | `docker compose up` | `http://localhost:8000` または `http://<ip>:80/keiba/` |
| uvicorn 直接 | `uvicorn api.app:app --reload` | `http://localhost:8000` |

### 10.2 ダッシュボード画面構成 (`/`)

```
┌─────────────────────────────────────────────────────────┐
│  🏇 ML-AutoPilot Keiba                                  │
│  マルチエージェント競馬AI予測システム                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ■ エージェント状態                                        │
│  ┌──────────────┐                                        │
│  │  🎖️ Orchestrator (将軍) │                             │
│  └───────┬──────┘                                        │
│          ↓                                               │
│  ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐                  │
│  │RDA ││SMA ││DOA ││FEA ││MTEA││POA │                  │
│  │📋  ││🌐  ││🧹  ││⚙️  ││🧠  ││📊  │                  │
│  └────┘└────┘└────┘└────┘└────┘└────┘                  │
│  各カードに: 名前 / 役割 / 状態(✅🔄❌⏸️) / 詳細           │
│                                                          │
│  ■ モデル評価サマリー                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                │
│  │ Accuracy │ │ F1 Score │ │ ROC AUC  │                │
│  │  87.3%   │ │  67.3%   │ │  87.9%   │                │
│  └──────────┘ └──────────┘ └──────────┘                │
│                                                          │
│  ■ 特徴量重要度 TOP10                                     │
│  odds          ████████████████████ 523                  │
│  horse_number  ██████████████ 371                        │
│  weight        ███████████ 298                           │
│  ...                                                     │
│                                                          │
│  ■ レース予測結果                                         │
│  ┌───────────────────────────────────────┐              │
│  │ 🏆 第XX回 ○○記念  📅 2026-XX-XX      │              │
│  │ 📍 東京  📏 芝2000m                   │              │
│  ├────┬────┬──────┬────┬──────┬─────────┤              │
│  │予測│馬番│馬名  │騎手│確率  │評価     │              │
│  │ 1  │ 3  │○○○○│△△ │72.3% │◎ 本命  │              │
│  │ 2  │ 7  │□□□□│▽▽ │55.1% │○ 対抗  │              │
│  │ 3  │ 1  │◇◇◇◇│×× │41.2% │▲ 単穴  │              │
│  └────┴────┴──────┴────┴──────┴─────────┘              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 10.3 テンプレートとスタイル

| ファイル | 役割 |
|---------|------|
| `templates/index.html` | Jinja2 テンプレート (ダッシュボード全体) |
| `static/style.css` | スタイルシート |

### 10.4 テンプレートに渡されるデータ

| 変数 | 型 | 内容 |
|------|-----|------|
| `data` | `dict \| None` | `predictions.json` の内容 |
| `data.model_evaluation` | `dict` | `{accuracy, f1_score, roc_auc}` |
| `data.feature_importance` | `dict` | `{feature_name: importance_score, ...}` TOP10 |
| `data.races` | `list[dict]` | レースごとの予測結果 |
| `data.mlflow_run_id` | `str \| None` | MLflow の run ID |
| `agent_status` | `dict` | 全エージェントの状態 `{name: {status, detail}}` |

### 10.5 予測が無い場合

`predictions.json` が存在しない場合、「予測データがありません」メッセージと
`python commander.py` の実行方法を案内する画面が表示される。

## 11. 予測結果出力フォーマット

### 11.1 出力ファイル

| ファイル | パス | 形式 |
|---------|------|------|
| 予測結果 | `data/processed/predictions.json` | JSON |

### 11.2 predictions.json スキーマ

```json
{
  "model_evaluation": {
    "accuracy": 0.8731,
    "f1_score": 0.6731,
    "roc_auc": 0.8785
  },
  "feature_importance": {
    "odds": 523,
    "horse_number": 371,
    "weight": 298
  },
  "mlflow_run_id": "b7e991fafc7e4a1...",
  "races": [
    {
      "race_id": "202606020611",
      "race_name": "第XX回○○賞",
      "date": "2026-03-15",
      "racecourse": "東京",
      "distance": 2000,
      "surface": "芝",
      "grade": "G1",
      "predictions": [
        {
          "predicted_rank": 1,
          "horse_number": 3,
          "horse_name": "ディープインパクト",
          "jockey": "武豊",
          "odds": 5.2,
          "top3_probability": 72.3,
          "recommendation": "◎ 本命"
        }
      ]
    }
  ]
}
```

### 11.3 推奨印 (recommendation) ルール

| 3着内確率 | 印 | 意味 |
|----------|-----|------|
| ≥ 70% | ◎ 本命 | 最も3着内に入る確率が高い |
| ≥ 55% | ○ 対抗 | 本命に次ぐ有力候補 |
| ≥ 40% | ▲ 単穴 | 穴馬として注目 |
| ≥ 25% | △ 連下 | 連複の相手候補 |
| < 25% | ☆ 穴馬 | 大穴候補 |

### 11.4 出力タイミング

POA (予測出力エージェント) がパイプラインの最終フェーズで自動生成する。
ファイルは上書きされるため、履歴は MLflow の run artifacts で管理する。

## 12. API リファレンス

### 12.1 予測 API (FastAPI)

| メソッド | パス | 説明 |
|---------|------|------|
| `GET` | `/` | Web ダッシュボード |
| `GET` | `/api/predictions` | 最新の予測結果 JSON |
| `GET` | `/api/status` | エージェント状態 + 最新ログ |
| `POST` | `/api/command` | 将軍にコマンド送信 |
| `POST` | `/api/requirements` | 要件を指定して予測実行 |
| `POST` | `/api/predict` | **MLflow モデルで特徴量 → 予測** |
| `GET` | `/api/mlflow/model-info` | 登録モデル情報 |
| `GET` | `/api/mlflow/history` | 学習履歴一覧 |

### 12.2 /api/predict リクエスト例

```json
POST /api/predict
{
  "features": [
    {"horse_number": 3, "age": 4, "weight": 480, "odds": 5.2, ...},
    {"horse_number": 7, "age": 3, "weight": 456, "odds": 12.8, ...}
  ],
  "model_name": "keiba-lgbm"
}
```

### 12.3 /api/predict レスポンス例

```json
{
  "status": "success",
  "model_info": {
    "run_id": "b7e991fafc7e...",
    "metrics": {"accuracy": 0.8731, "f1_score": 0.6731, "roc_auc": 0.8785}
  },
  "results": [
    {"index": 0, "prediction": 1, "probability": 0.7234, "recommendation": "◎ 本命"},
    {"index": 1, "prediction": 0, "probability": 0.3156, "recommendation": "△ 連下"}
  ]
}
```

### 12.4 /api/command リクエスト例

```json
POST /api/command
{
  "command": "予測して"
}
```

### 12.5 /api/requirements リクエスト例

```json
POST /api/requirements
{
  "requirements": ["芝のみ", "オッズ除外", "チューニング100回"]
}
```

### 12.6 /api/requirement-examples レスポンス例

```json
GET /api/requirement-examples
{
  "examples": [
    {"text": "予測して", "description": "デフォルト設定でフルパイプラインを実行"},
    {"text": "1着予測に変更して実行", "description": "ターゲットを1着に変更して予測"},
    {"text": "精度は？", "description": "モデル評価結果を表示"}
  ]
}
```

## 13. デプロイアーキテクチャ

### 13.1 Docker Compose 構成

```
mlflow_server/
├── docker-compose.yml     Docker Compose 定義
├── Dockerfile.api         Keiba FastAPI コンテナ
├── setup.sh               セットアップスクリプト
└── nginx/
    ├── nginx.conf          リバースプロキシ設定
    ├── htpasswd            Basic認証ファイル
    └── certs/              SSL証明書 (オプション)
```

### 13.2 サービス構成

```
┌─────────────────────────────────────────────────┐
│                   nginx (:80/:443)               │
│              Basic認証 + リバースプロキシ           │
├──────────┬──────────────────────────────────────┤
│          │                                       │
│  /       │  /keiba/*                             │
│  ↓       │  ↓                                    │
│  mlflow  │  keiba-api                            │
│  (:5000) │  (:8000)                              │
│          │                                       │
│  MLflow  │  FastAPI                              │
│  Tracking│  予測API + ダッシュボード               │
│  Server  │                                       │
│  SQLite  │  ┌──────────┐                         │
│  +       │  │data/     │ (ボリューム共有)          │
│  Artifacts│  │models/   │                         │
│          │  │config/   │                         │
│          │  └──────────┘                         │
└──────────┴──────────────────────────────────────┘
```

### 13.3 サービス詳細

| サービス | イメージ | ポート | 役割 |
|---------|---------|--------|------|
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.12.2` | 5000 | MLflow Tracking Server (SQLite + ファイルアーティファクト) |
| `nginx` | `nginx:alpine` | 80, 443 | リバースプロキシ (Basic認証, SSL終端) |
| `keiba-api` | カスタムビルド (`Dockerfile.api`) | 8000 | FastAPI 予測API + Web ダッシュボード |

### 13.4 ボリューム

| ボリューム | マウント先 | 内容 |
|-----------|-----------|------|
| `mlflow_data` | `/mlflow` | MLflow DB + アーティファクト |
| `../data` | `/app/data` | 学習/予測データ |
| `../models` | `/app/models` | ローカルモデルファイル |
| `../config` | `/app/config` | settings.yaml |

### 13.5 セットアップ手順

```bash
cd mlflow_server/
chmod +x setup.sh
./setup.sh
```

setup.sh が自動で以下を実行:
1. Basic認証パスワード生成 (`htpasswd`)
2. SSL証明書ディレクトリ準備 (自己署名証明書の生成オプション)
3. Docker Compose ビルド・起動
4. MLflow サーバーの起動確認

### 13.6 外部アクセスURL一覧

| リソース | ローカル | 外部 (nginx経由) |
|---------|---------|----------------|
| MLflow UI | `http://localhost:5000` | `http://<ip>:80/` |
| 予測ダッシュボード | `http://localhost:8000` | `http://<ip>:80/keiba/` |
| 予測 API | `http://localhost:8000/api/` | `http://<ip>:80/keiba/api/` |
| MLflow モデル予測 | `http://localhost:8000/api/predict` | `http://<ip>:80/keiba/api/predict` |
| 学習履歴 | `http://localhost:8000/api/mlflow/history` | `http://<ip>:80/keiba/api/mlflow/history` |

### 13.7 セキュリティ

| 項目 | 実装 |
|------|------|
| 認証 | Nginx Basic認証 (`htpasswd`) |
| TLS | SSL証明書設定 (Let's Encrypt or 自己署名) |
| ネットワーク分離 | Docker bridge network (`keiba-net`) |
| 環境変数 | `.env` に機密情報を分離 |

## 14. スクレイピング仕様

### 14.1 対応ページ

| ページ | URL パターン | パーサー | 認証 |
|--------|-------------|---------|------|
| レース結果 | `db.netkeiba.com/race/{race_id}/` | `RaceResultParser` | 不要 |
| 出馬表 | `race.netkeiba.com/race/shutuba.html?race_id={id}` | `RaceCardParser` | 不要 |
| 馬情報 | `db.netkeiba.com/horse/{horse_id}/` | `HorseParser` | 不要 |
| 日別一覧 | `db.netkeiba.com/race/list/{YYYYMMDD}/` | `RaceListParser` | 不要 |
| タイム指数 | `race.netkeiba.com/race/speed.html?race_id={id}` | `SpeedIndexParser` | **要** |
| 馬柱 | `race.netkeiba.com/race/shutuba_past.html?race_id={id}` | `ShutubaPastParser` | 不要 |

### 14.2 耐久設計

- **SelectorChain**: 1つのデータに対して複数の CSS セレクタを優先順に試行
- **エンコーディング**: `db.netkeiba.com` = EUC-JP, `race.netkeiba.com` = EUC-JP
- **レート制限**: `request_interval_sec` (デフォルト 1.0秒)
- **キャッシュ**: URL ベースの MD5 キー、認証時は `_auth` サフィックス
- **認証**: `.env` から `netkeiba_id` / `netkeiba_pw` を読み込み、セッション Cookie で管理

## 15. 設定リファレンス

### config/settings.yaml

| セクション | キー | 説明 | デフォルト |
|-----------|------|------|-----------|
| `project` | `name` | プロジェクト名 | ML-AutoPilot Keiba |
| `data_source` | `base_url` | データソースURL | https://db.netkeiba.com/ |
| `data_source` | `request_interval_sec` | リクエスト間隔 | 1.0 |
| `paths` | `models` | モデル保存先 | models |
| `model` | `target_variable` | 予測対象 | finish_position_top3 |
| `model` | `n_optuna_trials` | チューニング回数 | 50 |
| `model` | `test_size` | 検証データ比率 | 0.2 |
| `mlflow` | `tracking_uri` | MLflow サーバー URI | http://localhost:5000 |
| `mlflow` | `experiment_name` | 実験名 | keiba-prediction |
| `mlflow` | `registered_model_name` | モデル登録名 | keiba-lgbm |
| `api` | `port` | API ポート | 8000 |

### .env (認証情報)

| キー | 説明 |
|------|------|
| `netkeiba_id` | netkeiba.com ログインID |
| `netkeiba_pw` | netkeiba.com パスワード |

---

*このファイルはシステム起動時・コマンド実行時に `utils/spec_gen.py` により自動更新されます。*
*手動編集は次回の自動更新で上書きされます。*
