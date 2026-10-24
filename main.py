"""
keiba-vpn — Web Server + Data Platform

ConoHa VPN 上にデプロイする競馬データ基盤サーバー。
スクレイピング、ML パイプライン、データ閲覧、分析可視化を提供。

使い方:
  python main.py                # デフォルト (0.0.0.0:<config/ports.env の現環境 FastAPI ポート>)
  python main.py --port 9000    # ポート指定
  python main.py --host 127.0.0.1  # ローカルのみ
"""

from __future__ import annotations

import argparse
import os
import sys


def run_server(host: str = "0.0.0.0", port: int | None = None,
               workers: int = 1, reload: bool = True):
    import logging

    import uvicorn

    from src.config.ports import port as _service_port
    from src.utils.keiba_logging import apply_root_logging_basic, build_uvicorn_log_config

    port = port or _service_port("fastapi")

    # ログ・libc のローカル時刻を日本時間に揃える（Docker / stg が UTC のときの保険）
    os.environ.setdefault("TZ", "Asia/Tokyo")

    # uvicorn 以外（起動前の import など）向けに root を整える
    apply_root_logging_basic()
    _lg = logging.getLogger("main")
    log_cfg = build_uvicorn_log_config()
    mode = "development (reload)" if reload else f"production ({workers} workers)"
    _lg.info("keiba-vpn server starting: http://%s:%s  [%s]", host, port, mode)
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        workers=1 if reload else workers,
        reload=reload,
        # logs/・data/ など書き込みの多いディレクトリを監視すると再起動ループになるためコード類に限定
        reload_dirs=[os.path.join(os.getcwd(), d) for d in ("src", "templates", "static")] if reload else None,
        log_config=log_cfg,
        log_level="info",
        timeout_keep_alive=120,
        limit_concurrency=100,
    )


def main():
    parser = argparse.ArgumentParser(description="keiba-vpn — Data Platform Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None,
                        help="Bind port (default: config/ports.env の現在の KEIBA_ENV 用 FastAPI ポート)")
    parser.add_argument("--workers", type=int, default=4, help="Worker count for production (default: 4)")
    parser.add_argument("--prod", action="store_true", help="Production mode (multi-worker, no reload)")
    parser.add_argument(
        "--flask-api",
        action="store_true",
        help="Flask /api/v1 REST API のみ起動（仕様準拠 API サーバー）",
    )
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from src.config.deployment import require_explicit_env
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    require_explicit_env()

    if args.flask_api:
        from src.api.flask_app import main as flask_main
        flask_main()
        return

    run_server(host=args.host, port=args.port,
               workers=args.workers, reload=not args.prod)


if __name__ == "__main__":
    main()
