.PHONY: setup-dev setup-stg setup-prod \
	db-up db-down db-migrate db-reset \
	stg-up stg-down stg-migrate stg-reset \
	help

help:
	@echo "使い方: make <target>"
	@echo ""
	@echo "  [dev]"
	@echo "  setup-dev    .env コピー + Docker 起動 + DB マイグレーション (初回)"
	@echo "  db-up        Docker コンテナ起動 (PostgreSQL:5432 / Redis:6379)"
	@echo "  db-down      Docker コンテナ停止"
	@echo "  db-migrate   alembic upgrade head (dev)"
	@echo "  db-reset     ボリューム削除 → 再作成 → マイグレーション"
	@echo ""
	@echo "  [stg]"
	@echo "  setup-stg    .env.stg コピー + Docker 起動 + DB マイグレーション (初回)"
	@echo "  stg-up       Docker コンテナ起動 (PostgreSQL:5433 / Redis:6380)"
	@echo "  stg-down     Docker コンテナ停止"
	@echo "  stg-migrate  alembic upgrade head (stg)"
	@echo "  stg-reset    ボリューム削除 → 再作成 → マイグレーション"
	@echo ""
	@echo "  [prod (VPS)]"
	@echo "  setup-prod   PostgreSQL/Redis確認 + DB作成 + マイグレーション + cron"

# ─── dev ───────────────────────────────────────────────────────────────────────

setup-dev:
	@bash scripts/setup_dev.sh

db-up:
	docker compose -f docker-compose.dev.yml up -d
	@echo "Waiting for PostgreSQL..."
	@docker compose -f docker-compose.dev.yml exec -T postgres \
		sh -c 'until pg_isready -U keiba_user -d keiba_db -q; do sleep 1; done'
	@echo "PostgreSQL ready."

db-down:
	docker compose -f docker-compose.dev.yml down

db-migrate:
	DATABASE_URL=postgresql+psycopg://keiba_user:keiba_pass@localhost:5432/keiba_db \
		alembic upgrade head

db-reset:
	docker compose -f docker-compose.dev.yml down -v
	docker compose -f docker-compose.dev.yml up -d
	@docker compose -f docker-compose.dev.yml exec -T postgres \
		sh -c 'until pg_isready -U keiba_user -d keiba_db -q; do sleep 1; done'
	DATABASE_URL=postgresql+psycopg://keiba_user:keiba_pass@localhost:5432/keiba_db \
		alembic upgrade head
	@echo "DB reset complete."

# ─── stg ───────────────────────────────────────────────────────────────────────

setup-stg:
	@bash scripts/setup_stg.sh

stg-up:
	docker compose -f docker-compose.stg.yml up -d
	@echo "Waiting for PostgreSQL (stg)..."
	@docker compose -f docker-compose.stg.yml exec -T postgres-stg \
		sh -c 'until pg_isready -U keiba_user -d keiba_db_stg -q; do sleep 1; done'
	@echo "PostgreSQL (stg) ready."

stg-down:
	docker compose -f docker-compose.stg.yml down

stg-migrate:
	DATABASE_URL=postgresql+psycopg://keiba_user:keiba_pass@localhost:5433/keiba_db_stg \
		alembic upgrade head

stg-reset:
	docker compose -f docker-compose.stg.yml down -v
	docker compose -f docker-compose.stg.yml up -d
	@docker compose -f docker-compose.stg.yml exec -T postgres-stg \
		sh -c 'until pg_isready -U keiba_user -d keiba_db_stg -q; do sleep 1; done'
	DATABASE_URL=postgresql+psycopg://keiba_user:keiba_pass@localhost:5433/keiba_db_stg \
		alembic upgrade head
	@echo "STG DB reset complete."

# ─── prod (VPS) ────────────────────────────────────────────────────────────────

setup-prod:
	@bash scripts/setup_prod.sh
