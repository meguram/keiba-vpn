.PHONY: setup-dev db-up db-down db-migrate db-reset help

help:
	@echo "Dev environment targets:"
	@echo "  setup-dev   - .env コピー + Docker 起動 + DB マイグレーション (初回)"
	@echo "  db-up       - Docker コンテナ起動のみ"
	@echo "  db-down     - Docker コンテナ停止"
	@echo "  db-migrate  - alembic upgrade head"
	@echo "  db-reset    - コンテナ・ボリューム削除 → 再作成 → マイグレーション"

setup-dev:
	@bash scripts/setup_dev.sh

db-up:
	docker compose -f docker-compose.dev.yml up -d
	@echo "Waiting for PostgreSQL to be ready..."
	@docker compose -f docker-compose.dev.yml exec postgres sh -c \
		'until pg_isready -U keiba_user -d keiba_db; do sleep 1; done'

db-down:
	docker compose -f docker-compose.dev.yml down

db-migrate:
	DATABASE_URL=postgresql+psycopg://keiba_user:keiba_pass@localhost:5432/keiba_db \
		alembic upgrade head

db-reset:
	docker compose -f docker-compose.dev.yml down -v
	docker compose -f docker-compose.dev.yml up -d
	@echo "Waiting for PostgreSQL to be ready..."
	@docker compose -f docker-compose.dev.yml exec postgres sh -c \
		'until pg_isready -U keiba_user -d keiba_db; do sleep 1; done'
	DATABASE_URL=postgresql+psycopg://keiba_user:keiba_pass@localhost:5432/keiba_db \
		alembic upgrade head
	@echo "DB reset complete."
