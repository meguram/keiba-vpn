"""ノートブック内にインタラクティブ HTML を埋め込むユーティリティ。"""

from __future__ import annotations

import atexit
import importlib.util
import socket
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent
_PREVIEW_PORT = 18765
_server: ThreadingHTTPServer | None = None
_server_lock = threading.Lock()


def load_build_script(module_name: str) -> object:
    path = NOTEBOOK_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def notebook_rel_path(html_path: Path) -> str:
    """ノートブックから参照できる相対パス。"""
    return html_path.resolve().relative_to(NOTEBOOK_DIR).as_posix()


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _ensure_preview_server() -> str:
    """megu_index 配下を配信する HTTP サーバを起動し、ベース URL を返す。"""
    global _server
    with _server_lock:
        if _server is not None:
            return f"http://127.0.0.1:{_server.server_address[1]}"

        handler = partial(SimpleHTTPRequestHandler, directory=str(NOTEBOOK_DIR))
        for port in range(_PREVIEW_PORT, _PREVIEW_PORT + 20):
            if not _port_available(port):
                continue
            try:
                httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
            except OSError:
                continue
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            _server = httpd
            atexit.register(httpd.shutdown)
            return f"http://127.0.0.1:{port}"

    raise RuntimeError("megu_index HTML プレビュー用ポートを確保できませんでした")


def preview_url(html_path: Path) -> str:
    """HTTP プレビュー URL を組み立てる。"""
    resolved = html_path.resolve()
    rel = resolved.relative_to(NOTEBOOK_DIR).as_posix()
    return f"{_ensure_preview_server()}/{rel}"


def build_iframe_markup(html_path: Path, height: int = 680) -> str:
    """iframe HTML 文字列を生成（ノートブック本体にスタイルを漏らさない）。"""
    html_path = html_path.resolve()
    if not html_path.is_file():
        raise FileNotFoundError(f"HTML not found: {html_path}")

    url = preview_url(html_path)
    return (
        f'<iframe src="{url}" width="100%" height="{height}" '
        'style="border:1px solid var(--vscode-panel-border,#d1d5db);'
        'border-radius:8px;width:100%;display:block;" '
        'loading="lazy"></iframe>'
    )


def display_interactive_html(html_path: Path, height: int = 680, title: str | None = None) -> None:
    """セル出力直下に iframe で HTML を表示（ノートブックテーマは変更しない）。"""
    from IPython.display import HTML, IFrame, display

    resolved = Path(html_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"HTML not found: {resolved}")

    rel = notebook_rel_path(resolved)
    url = preview_url(resolved)

    if title:
        display(HTML(f"<p><em>{title}</em></p>"))

    # 全文 HTML の isolated 埋め込みは Cursor でスタイル漏れ→背景が真っ白になるため使わない
    display(IFrame(src=url, width="100%", height=height))

    display(
        HTML(
            f"<p><small>表示されない場合: エクスプローラで <code>{rel}</code> を開き "
            f'<a href="{url}" target="_blank" rel="noopener">Live Preview / ブラウザ</a> '
            f"で確認してください。</small></p>"
        )
    )

    print(f"表示: {resolved}")
    print(f"  相対パス: {rel}")
    print(f"  プレビュー URL: {url}")
