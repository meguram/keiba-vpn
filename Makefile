.PHONY: setup-dev setup-stg setup-prod \
	db-up db-down db-migrate db-reset \
	stg-migrate \
	test test-frontend test-all \
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
	@echo "  [stg] ネイティブ PostgreSQL 14 (:5432) 前提。scripts/server/setup_stg.sh 参照"
	@echo "  setup-stg    ネイティブPostgreSQL起動確認 + DB作成 + マイグレーション + ETL"
	@echo "  stg-migrate  alembic upgrade head (stg, DATABASE_URL は .env.stg 参照)"
	@echo ""
	@echo "  [prod (VPS)]"
	@echo "  setup-prod   PostgreSQL/Redis確認 + DB作成 + マイグレーション + cron"
	@echo ""
	@echo "  [test]"
	@echo "  test          pytest 一括実行（CIと同じ除外設定。DB未起動でも大半は動く）"
	@echo "  test-frontend frontend/ の lint + build"
	@echo "  test-all      test + test-frontend"

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

# ─── stg（ネイティブ PostgreSQL 14 :5432 前提。Docker は不要） ───────────────────

setup-stg:
	@bash scripts/server/setup_stg.sh

stg-migrate:
	DATABASE_URL=postgresql+psycopg://keiba_user:keiba_pass@localhost:5432/keiba_db_stg \
		alembic upgrade head

# ─── prod (VPS) ────────────────────────────────────────────────────────────────

setup-prod:
	@bash scripts/setup_prod.sh

# ─── test ──────────────────────────────────────────────────────────────────────
# CI（.github/workflows/ci.yml）と同じ除外設定・実行順序を local でも再現する。
# DB/Redis 未起動でも大半のテストは動くが、DB依存テストは失敗/エラーになる
# （事前に `make db-up && make db-migrate` を推奨）。

test:
	python3 -m src.scripts.ci.check_no_shuffle
	python3 -m pytest tests/pipeline/test_build_rank_target.py -v --tb=short
	python3 -m pytest tests/ --ignore=tests/scraper/manual --ignore=tests/research/manual --tb=short -ra

test-frontend:
	cd frontend && npm run lint && npm run build

test-all: test test-frontend
