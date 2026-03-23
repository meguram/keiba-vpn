"""
keiba-vpn — Web Server + Data Platform

ConoHa VPN 上にデプロイする競馬データ基盤サーバー。
スクレイピング、ML パイプライン、データ閲覧、分析可視化を提供。

使い方:
  python main.py                # デフォルト (0.0.0.0:8000)
  python main.py --port 9000    # ポート指定
  python main.py --host 127.0.0.1  # ローカルのみ
"""

from __future__ import annotations

import argparse
import os
import sys


def run_server(host: str = "0.0.0.0", port: int = 8000,
               workers: int = 1, reload: bool = True):
    import uvicorn
    mode = "development (reload)" if reload else f"production ({workers} workers)"
    print(f"keiba-vpn server starting: http://{host}:{port}  [{mode}]")
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        workers=1 if reload else workers,
        reload=reload,
        log_level="info",
        timeout_keep_alive=120,
        limit_concurrency=100,
    )


def main():
    parser = argparse.ArgumentParser(description="keiba-vpn — Data Platform Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--workers", type=int, default=4, help="Worker count for production (default: 4)")
    parser.add_argument("--prod", action="store_true", help="Production mode (multi-worker, no reload)")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_server(host=args.host, port=args.port,
               workers=args.workers, reload=not args.prod)


if __name__ == "__main__":
    main()
