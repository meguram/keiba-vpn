# keiba-vpn — 競馬データ基盤サーバー

ML-AutoPilot Keiba のデータ基盤。スクレイピング、ML パイプライン、データ閲覧、分析可視化を提供する Web サーバー。

## セットアップ (ConoHa VPN)

```bash
# 1. リポジトリをクローン
git clone <repo-url> keiba-vpn
cd keiba-vpn

# 2. Python仮想環境を作成
python3 -m venv .venv
source .venv/bin/activate

# 3. 依存パッケージをインストール
pip install -r requirements.txt

# 4. 環境変数を設定
cp .env.example .env
# .env を編集して認証情報を入力

# 5. サーバーを起動
python main.py --port 8000
```

## ディレクトリ構成

すべての Python コードは **`src/`** 配下に集約し、役割ごとにサブパッケージで分割している。
シェルスクリプト・データ・テスト・ノートブックはトップレベルに残す。

```
keiba-vpn/
├── src/                  # Python コード一式（importは src.X.Y 形式）
│   ├── api/                   # FastAPI アプリケーション (app.py / auth.py)
│   ├── scraper/               # スクレイピング (netkeiba, SmartRC, JRA)
│   ├── pipeline/              # ML パイプライン (LightGBM, XGBoost, CatBoost)
│   │   ├── features/               # 特徴量ストア・レイアウト・ビルダー・統計
│   │   ├── models/                 # 学習・予測子 (trainer, encoder, *_predictor)
│   │   ├── inference/              # 推論実行 (race_day, betting, composite_optimizer)
│   │   └── build_*.py / cli.py     # CLI（python -m src.pipeline.build_X）
│   ├── research/              # リサーチ層
│   │   ├── pedigree/               # 血統・種牡馬分析（bloodline / sire / stage5_3 等）
│   │   ├── genes/                  # 遺伝子マーカー（myostatin / performance_genes）
│   │   ├── race/                   # コース・トラック・レース品質
│   │   └── scripts/                # キュー投入・試行系 CLI
│   ├── scripts/               # 運用 Python CLI
│   │   ├── scraping/               # 各種スクレイピング
│   │   ├── data/                   # backfill / build / sync / upload 等
│   │   ├── docs/                   # ドキュメント生成 (HTML 埋め込み)
│   │   └── maintenance/            # 整備・可視化
│   └── utils/                 # ユーティリティ (MLflow, ロギング等)
├── scripts/              # 運用シェルスクリプト
│   ├── server/                # サーバ起動・監視 (restart / watchdog)
│   └── cron/                  # cron / 定期実行
├── templates/            # Jinja2 HTML テンプレート
├── static/               # CSS / ベンダー資産
├── notebooks/            # Jupyter ノートブック（と伴走スクリプト）
│   ├── pedigree/              # 血統系探索 ipynb
│   ├── feature_engineering/   # FE 探索 (llm_research.ipynb + run_*.py + _run_output)
│   └── modeling/              # モデリング探索 ipynb
├── models/               # 学習済みモデル（アーティファクト）
├── data/                 # データ (生・メタ・特徴量・知識DB 等)
├── config/               # 設定ファイル
├── docs/                 # ドキュメント
│   ├── ARCHITECTURE.html / DATASOURCES.html / cost.html
│   ├── modeling/              # 仕様書（設計の正）
│   └── html/                  # 参照用 HTML
├── tests/                # ユニットテスト
├── mlflow/               # MLflow（data/=Docker DB+artifacts, runs/=ローカルフォールバック, server/=Docker デプロイ）
├── main.py               # サーバエントリポイント（src.api.app:app を起動）
└── requirements.txt
```

Python の CLI は `python -m src.<pkg>.<mod>` で呼び出す。シェル用ヘルパは
`scripts/server/restart_server.sh`・`scripts/cron/update_jockey_trainer_stats.sh`
などを引き続き使う（中身は src/ の新パスに更新済み）。

## ページ一覧

| パス | 機能 |
|------|------|
| `/` | ダッシュボード (予測結果表示) |
| `/monitor` | GCS スクレイピングモニター |
| `/data-viewer` | 生データビューア |
| `/agents` | エージェントステータス (keiba Agent 接続時) |
| `/race/{race_id}` | レース詳細 |
| `/betting` | 馬券最適化 (Kelly Criterion) |
| `/bloodline` | 血統 × 距離適性研究 |
| `/course-bloodline` | コース特性 × 血統適性 |
| `/myostatin` | ミオスタチン遺伝子検索 |
| `/growth-curve` | 成長曲線（馬体重・パフォーマンス推移） |

## keiba Agent との連携

keiba Agent (ローカル) から本サーバーに接続して操作可能:

```bash
# keiba 側の .env に VPN_API_URL を設定
VPN_API_URL=http://<conoha-vpn-ip>:8000

# keiba 側から操作
cd ../keiba
python commander.py
```

Agent は HTTP API 経由で本サーバーのスクレイピング・パイプラインを制御する。

## 競走馬遺伝子研究

競走馬のパフォーマンスに関連する遺伝子マーカーのナレッジベースを構築。

### 対象遺伝子
- **MSTN (Myostatin)** - 筋量制御、距離適性（220頭以上の種牡馬データ）
- **DMRT3** - 歩様遺伝子（ギャロップ vs 速歩）
- **PDK4** - エネルギー代謝（グルコース vs 脂肪酸酸化）
- **COX4I2** - ミトコンドリア機能（持久力）
- **BIEC2-808543** - 距離適性SNPマーカー

### 使用方法

```bash
# ミオスタチン遺伝子検索
python3 -m src.research.genes.myostatin lookup ディープインパクト
python3 -m src.research.genes.myostatin predict ディープインパクト キングカメハメハ --distance 2000

# その他のパフォーマンス遺伝子
python3 -m src.research.genes.performance_genes combined --mstn CT --pdk4 GA --distance 1800
```

詳細: [docs/html/PERFORMANCE_GENES.html](docs/html/PERFORMANCE_GENES.html)

## 関連ドキュメント

- [docs/html/ARCHITECTURE.html](docs/html/ARCHITECTURE.html) — システム全体のアーキテクチャ
- [docs/html/DATASOURCES.html](docs/html/DATASOURCES.html) — データソース仕様
- [docs/html/modeling/](docs/html/modeling/) — モデリング・レーティング・データセット設計仕様
- [docs/html/cost.html](docs/html/cost.html) — GCS コスト分析
