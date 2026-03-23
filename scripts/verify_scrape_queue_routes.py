#!/usr/bin/env python3
"""
キューUI・タスクカタログ用の API ルートがアプリに載っているか確認する。

  python scripts/verify_scrape_queue_routes.py
  BASE_URL=http://127.0.0.1:8000 python scripts/verify_scrape_queue_routes.py
"""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_PATHS = (
    "/api/scrape-queue/tasks",
    "/api/scrape-queue/add-job",
    "/api/scrape-queue/enqueue-period-horses",
    "/api/scrape-queue/enqueue-period-races",
)


def _http_get_code(url: str) -> int:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except OSError as e:
        print(f"接続失敗 {url}: {e}", file=sys.stderr)
        return -1


def check_app_object() -> list[str]:
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from api.app import app

    paths = {getattr(r, "path", "") for r in app.routes if getattr(r, "path", None)}
    return [p for p in REQUIRED_PATHS if p not in paths]


def check_http(base: str) -> list[str]:
    base = base.rstrip("/")
    problems: list[str] = []

    c = _http_get_code(base + "/api/scrape-queue/tasks")
    if c != 200:
        problems.append(f"GET /api/scrape-queue/tasks → HTTP {c}（200 であるべき）")

    # POST 専用: ルートがあれば GET は 405、無ければ 404
    c = _http_get_code(base + "/api/scrape-queue/add-job")
    if c == 404:
        problems.append("GET /api/scrape-queue/add-job → 404（ルート未登録。405 なら存在）")

    # enqueue-period-horses の GET dry_run は race_lists 全走査で遅い・タイムアウトしうるため HTTP 検証から除外

    return problems


def main() -> int:
    os.chdir(_ROOT)

    miss = check_app_object()
    if miss:
        print("アプリ定義にルート不足:", miss, file=sys.stderr)
        return 1
    print("アプリ定義: scrape-queue 必須ルート OK")

    base = os.environ.get("BASE_URL", "").strip()
    if not base:
        return 0

    probs = check_http(base)
    if probs:
        for p in probs:
            print(p, file=sys.stderr)
        print(
            "動いているプロセスが古いコードの可能性が高いです。uvicorn / サービスを再起動してください。",
            file=sys.stderr,
        )
        return 2
    print(f"HTTP {base}: キュー関連エンドポイント応答 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
