#!/usr/bin/env python3
"""フロント（Next.js）を止めている間だけ、同じポートで『メンテナンス中』の静的ページを HTTP 503 で返す小さなサーバ。

通常のメンテナンス表示は Next.js 側（/maintenance）。これは、フロントの再ビルド・入れ替えなどで Next.js 自体が止まるときの保険で、
手動で使う（自動反映ランナーは使わない: 切替スクリプトと同じポートを取り合うため）。標準ライブラリのみ。

  python3 scripts/ops/maintenance_holding_server.py --port 23200 [--seconds 600]   # Ctrl-C か指定秒数で終了
"""

from __future__ import annotations

import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).resolve().parents[2] / "static" / "maintenance.html"


def make_handler(page: bytes, retry_after: int):
    class H(BaseHTTPRequestHandler):
        def _send(self, body: bool) -> None:
            if self.path.startswith("/api/"):
                data = '{"maintenance": true, "message": "ただいまメンテナンス中です。"}'.encode("utf-8")
                ctype = "application/json; charset=utf-8"
            else:
                data, ctype = page, "text/html; charset=utf-8"
            self.send_response(503)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Retry-After", str(retry_after))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("X-Robots-Tag", "noindex")
            self.send_header("X-Keiba-Maintenance", "1")
            self.end_headers()
            if body:
                self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            self._send(True)

        def do_HEAD(self):  # noqa: N802
            self._send(False)

        def log_message(self, *a):  # noqa: D401
            pass

    return H


def serve(port: int, *, host: str = "0.0.0.0", seconds: float | None = None, retry_after: int = 60,
          page_path: Path = PAGE) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(page_path.read_bytes(), retry_after))
    if seconds:
        threading.Timer(seconds, srv.shutdown).start()
    return srv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--retry-after", type=int, default=60)
    a = ap.parse_args()
    srv = serve(a.port, host=a.host, seconds=a.seconds, retry_after=a.retry_after)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
