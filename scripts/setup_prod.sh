#!/usr/bin/env bash
# VPS (ConoHa Ubuntu) 本番環境セットアップ
#
# 実行順序:
#   1. PostgreSQL 15 / Redis 7 のインストール確認
#   2. DB / ユーザー作成
#   3. Python 依存パッケージインストール
#   4. .env 準備
#   5. Alembic マイグレーション
#   6. Cron ジョブ登録
#
# 使い方:
#   bash scripts/setup_prod.sh          # 全ステップ実行
#   bash scripts/setup_prod.sh --skip-packages  # パッケージインストールをスキップ
#   bash scripts/setup_prod.sh --db-only        # DB セットアップ + マイグレーションのみ
#   bash scripts/setup_prod.sh --migrate-only   # alembic upgrade head のみ
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

SKIP_PACKAGES=false
DB_ONLY=false
MIGRATE_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --skip-packages) SKIP_PACKAGES=true ;;
        --db-only)       DB_ONLY=true ;;
        --migrate-only)  MIGRATE_ONLY=true ;;
    esac
done

DB_USER="keiba_user"
DB_PASS="keiba_pass"   # 本番では .env の DATABASE_URL を直接指定してください
DB_NAME="keiba_db"
DB_URL="postgresql+psycopg://${DB_USER}:${DB_PASS}@localhost:5432/${DB_NAME}"

log() { echo "[setup-prod] $*"; }
die() { echo "[setup-prod] ERROR: $*" >&2; exit 1; }

# ── migrate-only ─────────────────────────────────────────────────────────────
if $MIGRATE_ONLY; then
    log "alembic upgrade head のみ実行..."
    if [ -f .env ]; then
        # shellcheck disable=SC1091
        set -a; source .env; set +a
    fi
    alembic upgrade head
    log "マイグレーション完了。"
    exit 0
fi

# ── 1. パッケージ確認 / インストール ──────────────────────────────────────────
if ! $SKIP_PACKAGES; then
    log "パッケージを確認します..."
    if ! command -v psql &>/dev/null; then
        log "PostgreSQL 15 をインストールします..."
        sudo apt-get update -qq
        sudo apt-get install -y postgresql-15 postgresql-client-15
        sudo systemctl enable postgresql
        sudo systemctl start postgresql
    else
        log "PostgreSQL: $(psql --version)"
    fi

    if ! command -v redis-cli &>/dev/null; then
        log "Redis をインストールします..."
        sudo apt-get update -qq
        sudo apt-get install -y redis-server
        sudo systemctl enable redis-server
        sudo systemctl start redis-server
    else
        log "Redis: $(redis-server --version | head -1)"
    fi
fi

if $DB_ONLY || ! $MIGRATE_ONLY; then
    # ── 2. DB / ユーザー作成 ────────────────────────────────────────────────────
    log "PostgreSQL DB / ユーザーを作成します..."
    sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" \
        | grep -q 1 || \
        sudo -u postgres psql -c "CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}';"

    sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" \
        | grep -q 1 || \
        sudo -u postgres psql -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"

    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};"
    # pgcrypto (gen_random_uuid) は alembic 001 で CREATE EXTENSION IF NOT EXISTS として実行されます
    sudo -u postgres psql -d "${DB_NAME}" -c "GRANT USAGE ON SCHEMA public TO ${DB_USER};"
    sudo -u postgres psql -d "${DB_NAME}" -c "GRANT CREATE ON SCHEMA public TO ${DB_USER};"
    log "DB 準備完了: ${DB_NAME} / ${DB_USER}"
fi

if $DB_ONLY; then
    log "--db-only 完了（マイグレーションはスキップ）。alembic upgrade head を別途実行してください。"
    exit 0
fi

# ── 3. Python 依存パッケージ ──────────────────────────────────────────────────
if ! $SKIP_PACKAGES; then
    if [ -x ".venv/bin/pip" ]; then
        log "pip install -r requirements.txt (.venv)..."
        .venv/bin/pip install -r requirements.txt -q
    elif command -v pip3 &>/dev/null; then
        log "pip3 install -r requirements.txt..."
        pip3 install -r requirements.txt -q
    else
        log "警告: pip が見つかりません。手動で requirements.txt をインストールしてください。"
    fi
fi

# ── 4. .env 準備 ────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    cp .env.example .env
    log ".env を .env.example からコピーしました。本番用の値を設定してください。"
    log "特に以下を変更してください:"
    log "  APP_SECRET_KEY, JWT_SECRET  (32文字以上のランダム文字列)"
    log "  DATABASE_URL               (上記で作成した接続文字列)"
    log "  DEV_PASSWORD, DEV_SECRET_KEY"
else
    log ".env は既に存在します（スキップ）。"
fi

# ── 5. Alembic マイグレーション ───────────────────────────────────────────────
log "DB マイグレーションを実行します..."
if [ -f .env ]; then
    # shellcheck disable=SC1091
    set -a; source .env; set +a
fi
: "${DATABASE_URL:=$DB_URL}"
DATABASE_URL="$DATABASE_URL" alembic upgrade head
log "マイグレーション完了。"

# ── 6. Cron ジョブ ────────────────────────────────────────────────────────────
if [ -x "scripts/cron/setup_all_cron.sh" ]; then
    log "Cron ジョブを登録します..."
    bash scripts/cron/setup_all_cron.sh
    log "Cron 登録完了。"
else
    log "scripts/cron/setup_all_cron.sh が見つかりません（cron はスキップ）。"
fi

echo ""
echo "✓ Prod セットアップ完了"
echo "  PostgreSQL : localhost:5432 (DB=${DB_NAME}, User=${DB_USER})"
echo "  Redis      : localhost:6379"
echo "  Flask 起動 : ./service_start --env prod"
echo "  状態確認   : ./service_start --status"
echo ""
echo "  ⚠️  .env の APP_SECRET_KEY / JWT_SECRET を本番用ランダム値に変更してください"
