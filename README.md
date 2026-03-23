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
