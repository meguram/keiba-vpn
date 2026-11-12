#!/usr/bin/env bash
# このPCで実行中・待機中のすべてのデータ取得と終了の見込み（読み取りのみ。終わったものは出さない）。
# 使い方: watch -n 5 bash scripts/dev/fetch_status.sh  仕組みと、取得を足すときの決まり: docs/operations/fetch_tasks.md
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR" && exec "${PYTHON:-/opt/conda/bin/python3}" -m src.scripts.scraping.fetch_status
