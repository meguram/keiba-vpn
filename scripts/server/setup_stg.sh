#!/usr/bin/env bash
# =================================================================
# setup_stg.sh — stg 環境初期セットアップ
# =================================================================
# 実行前に一度だけ行う初期作業:
#   1. PostgreSQL 14 クラスタ起動
#   2. keiba_user / keiba_db_stg 作成
#   3. Alembic マイグレーション（DB スキーマ構築）
#   4. stg DB ETL (GCS → DB データ投入)
#   5. stg サービス一括起動 (FastAPI, Flask, Next.js)
#
# Usage:
#   bash scripts/server/setup_stg.sh
#   bash scripts/server/setup_stg.sh --skip-etl   # ETLをスキップ
# =================================================================

set -euo pipefail
export TZ="${TZ:-Asia/Tokyo}"

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$(command -v python3)}"
SKIP_ETL=false

for arg in "$@"; do
  case "$arg" in
    --skip-etl) SKIP_ETL=true ;;
  esac
done

echo "=== stg セットアップ開始 ==="

# 1. PostgreSQL クラスタ起動
echo "[1/5] PostgreSQL 起動確認..."
if pg_isready -h localhost -p 5432 -q 2>/dev/null; then
  echo "  PostgreSQL :5432 — 既に稼働中"
else
  echo "  pg_ctlcluster 14 main start..."
  pg_ctlcluster 14 main start 2>&1 || { echo "  ERROR: PostgreSQL 起動失敗"; exit 1; }
  sleep 3
fi

# 2. DB ユーザー・DB 作成（冪等）
echo "[2/5] DB ユーザー・DB 作成..."
sudo -u postgres psql -c "CREATE USER keiba_user WITH PASSWORD 'keiba_pass' CREATEDB;" 2>&1 | grep -v "already exists" | head -3 || true
sudo -u postgres psql -c "CREATE DATABASE keiba_db_stg OWNER keiba_user;" 2>&1 | grep -v "already exists" | head -3 || true
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE keiba_db_stg TO keiba_user;" 2>&1 | head -2 || true

# 3. Alembic マイグレーション
echo "[3/5] Alembic マイグレーション..."
DATABASE_URL="postgresql+psycopg://keiba_user:keiba_pass@localhost:5432/keiba_db_stg" \
  "$PYTHON" -m alembic upgrade head 2>&1 | grep -E "INFO|ERROR" | head -5

# 4. ETL (GCS → DB)
if $SKIP_ETL; then
  echo "[4/5] ETL スキップ（--skip-etl）"
else
  echo "[4/5] stg DB ETL 実行（バックグラウンド）..."
  echo "  全2026年データを投入中... (15~30分かかります)"
  KEIBA_ENV=stg DATABASE_URL="postgresql+psycopg://keiba_user:keiba_pass@localhost:5432/keiba_db_stg" \
    nohup "$PYTHON" -m src.scripts.data.etl_stg_db --year 2026 --batch-size 30 \
    >> "$ROOT/logs/etl_stg_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
  echo "  ETL PID: $! (ログ: logs/etl_stg_*.log)"
fi

# 5. stg サービス起動
echo "[5/5] stg サービス起動..."
bash "$ROOT/scripts/server/service_start.sh" --env stg 2>&1

echo ""
echo "=== stg セットアップ完了 ==="
echo "  FastAPI  http://127.0.0.1:8000/"
echo "  Flask    http://127.0.0.1:5000/api/v1/health"
echo "  Next.js  http://127.0.0.1:3001/"
echo "  stg URL  https://meguai-stg.tcpexposer.com/"
