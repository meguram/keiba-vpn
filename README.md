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

```
keiba-vpn/
├── api/           # FastAPI アプリケーション
├── templates/     # Jinja2 HTML テンプレート
├── static/        # CSS
├── scraper/       # スクレイピング (netkeiba, SmartRC, JRA)
├── pipeline/      # ML パイプライン (LightGBM, XGBoost, CatBoost)
├── research/      # 血統・コース分析
├── models/        # 学習済みモデル
├── data/          # データ (知識DB, JRA馬場, 研究結果)
├── config/        # 設定ファイル
├── scripts/       # バッチスクレイピング・cron設定
├── utils/         # ユーティリティ (MLflow, etc.)
├── mlflow_server/ # MLflow Docker デプロイ
├── main.py        # エントリーポイント
└── requirements.txt
```

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
python3 -m research.myostatin lookup ディープインパクト
python3 -m research.myostatin predict ディープインパクト キングカメハメハ --distance 2000

# その他のパフォーマンス遺伝子
python3 -m research.performance_genes combined --mstn CT --pdk4 GA --distance 1800
```

詳細: [docs/PERFORMANCE_GENES.md](docs/PERFORMANCE_GENES.md)
