#!/usr/bin/env bash
# 取得の状況と終了の見込み（読み取りのみ）。使い方: watch -n 5 bash scripts/dev/fetch_status.sh
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR" && exec "${PYTHON:-/opt/conda/bin/python3}" -m src.scripts.scraping.fetch_status
