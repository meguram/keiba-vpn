#!/usr/bin/env bash
# Dev 環境セットアップ: .env 準備 → Docker 起動 → DB マイグレーション
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# ── 1. .env ──────────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    cp .env.example .env
    echo "[setup] .env を .env.example からコピーしました。必要に応じて値を編集してください。"
else
    echo "[setup] .env は既に存在します（スキップ）。"
fi

# ── 2. Docker ────────────────────────────────────────────────────────────────
echo "[setup] Docker コンテナを起動します..."
docker compose -f docker-compose.dev.yml up -d

echo "[setup] PostgreSQL が起動するまで待機します..."
for i in $(seq 1 30); do
    if docker compose -f docker-compose.dev.yml exec -T postgres \
            pg_isready -U keiba_user -d keiba_db -q 2>/dev/null; then
        echo "[setup] PostgreSQL 準備完了。"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "[setup] ERROR: PostgreSQL が 30 秒以内に起動しませんでした。" >&2
        exit 1
    fi
    sleep 1
done

# ── 3. Python 依存パッケージ（任意）──────────────────────────────────────────
if command -v pip &>/dev/null && [ ! -f .venv/pyvenv.cfg ] 2>/dev/null; then
    echo "[setup] pip install -r requirements.txt を実行します..."
    pip install -r requirements.txt -q
fi

# ── 4. Alembic マイグレーション ───────────────────────────────────────────────
echo "[setup] DB マイグレーションを実行します..."
DATABASE_URL=postgresql+psycopg://keiba_user:keiba_pass@localhost:5432/keiba_db \
    alembic upgrade head

echo ""
echo "✓ セットアップ完了"
echo "  PostgreSQL : localhost:5432 (DB=keiba_db, User=keiba_user, Pass=keiba_pass)"
echo "  Redis      : localhost:6379"
echo "  Flask 起動 : python -m src.api.flask_app  (または  make run)"
