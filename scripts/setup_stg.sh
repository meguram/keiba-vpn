#!/usr/bin/env bash
# Staging 環境セットアップ: .env.stg 準備 → Docker 起動 → DB マイグレーション
# dev と同時起動可能 (PostgreSQL :5433 / Redis :6380 / DB=keiba_db_stg)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

STG_ENV_FILE=".env.stg"
DB_URL="postgresql+psycopg://keiba_user:keiba_pass@localhost:5433/keiba_db_stg"

# ── 1. .env.stg ───────────────────────────────────────────────────────────────
if [ ! -f "$STG_ENV_FILE" ]; then
    cp .env.stg.example "$STG_ENV_FILE"
    echo "[setup-stg] $STG_ENV_FILE を .env.stg.example からコピーしました。値を編集してください。"
else
    echo "[setup-stg] $STG_ENV_FILE は既に存在します（スキップ）。"
fi

# ── 2. Docker ────────────────────────────────────────────────────────────────
echo "[setup-stg] Docker コンテナを起動します..."
docker compose -f docker-compose.stg.yml up -d

echo "[setup-stg] PostgreSQL (stg :5433) が起動するまで待機します..."
for i in $(seq 1 30); do
    if docker compose -f docker-compose.stg.yml exec -T postgres-stg \
            pg_isready -U keiba_user -d keiba_db_stg -q 2>/dev/null; then
        echo "[setup-stg] PostgreSQL 準備完了。"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "[setup-stg] ERROR: PostgreSQL が 30 秒以内に起動しませんでした。" >&2
        exit 1
    fi
    sleep 1
done

# ── 3. Alembic マイグレーション ───────────────────────────────────────────────
echo "[setup-stg] DB マイグレーションを実行します..."
DATABASE_URL="$DB_URL" alembic upgrade head

echo ""
echo "✓ Staging セットアップ完了"
echo "  PostgreSQL : localhost:5433 (DB=keiba_db_stg)"
echo "  Redis      : localhost:6380"
echo "  Flask 起動 : KEIBA_ENV=stg FLASK_PORT=5000 DATABASE_URL=$DB_URL python main.py --flask-api"
echo "  一括起動   : ./service_start --env stg"
