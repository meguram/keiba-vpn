"""「絶対回収率プラスにするマン」の会議の記録を、ブラウザで読むための小さな読み取り専用サーバ。

    python -m src.scripts.docs.council_viewer                       # 127.0.0.1:KEIBA_PORT_COUNCIL（config/ports.env）
    python -m src.scripts.docs.council_viewer --port 8090 --root /path/to/keiba-vpn

- 要求のたびに ``docs/operations/roi_plus_council/`` の Markdown を読み、HTML にして返す（キャッシュしない。常に最新）。
- 加えて ``docs/html/operations/final/`` の会議に関係する HTML（許可した名前だけ）をそのまま返す。
- 読むのは上の 2 か所だけ。``..``・絶対パス・シンボリックリンクで外に出る要求は 404。書き込みはしない。
- 認証が無いので、既定は 127.0.0.1 で待ち受ける（Cursor の Remote-SSH・ssh -L で手元へ転送して見る）。
- GCS・外部の網は読まない（費用 0 円）。
"""
from __future__ import annotations

import argparse
import html
import os
import re
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
COUNCIL_REL = Path("docs/operations/roi_plus_council")
HTML_REL = Path("docs/html/operations/final")
# URL はファイルの相対パスをそのまま映す（Markdown の相対リンクがそのまま効く）
COUNCIL_PREFIX = "/council/"
HTML_PREFIX = "/html/operations/final/"
HTML_FIXED = frozenset({"longshot_course_rules.html", "megu_cycles_50.html", "megu_why_below_100.html"})
HTML_PATTERN = re.compile(r"^rules_council_[A-Za-z0-9_\-]+\.html$")
RC_NAME = re.compile(r"^R(\d+)_(\d+)_.*\.md$")
DATE_RE = re.compile(r"20\d\d-\d\d-\d\d")
RC_NO = re.compile(r"RC-(\d+)")

# docs/html/operations/final/latest_strategy_tables.html と同じ CSS（明・暗）。外部の CDN は使わない
CSS = """
:root { --bg:#f7f8fb; --panel:#ffffff; --panel2:#eef1f7; --border:#d5dbe7; --text:#1b2333; --muted:#5b6578;
  --accent:#2f5fc4; --ok:#1d7f45; --warn:#9a6200; --bad:#b4232c; --code:#eef1f7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#0f172a; --panel:#15213c; --panel2:#1b2a4a; --border:#2a3a60;
  --text:#e6ecff; --muted:#9aa7c7; --accent:#7aa9ff; --ok:#4ade80; --warn:#fbbf24; --bad:#f87171; --code:#0b1428; } }
:root[data-theme="dark"] { --bg:#0f172a; --panel:#15213c; --panel2:#1b2a4a; --border:#2a3a60; --text:#e6ecff; --muted:#9aa7c7;
  --accent:#7aa9ff; --ok:#4ade80; --warn:#fbbf24; --bad:#f87171; --code:#0b1428; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.7 "Noto Sans JP",system-ui,sans-serif; }
main { max-width:1180px; margin:0 auto; padding:28px 16px 80px; }
a { color:var(--accent); }
h1 { font-size:24px; margin:0 0 6px; line-height:1.4; }
h2 { font-size:19px; margin:40px 0 6px; padding-top:16px; border-top:1px solid var(--border); }
h3 { font-size:16px; margin:22px 0 4px; }
p, li { overflow-wrap:anywhere; }
.sub { color:var(--muted); font-size:13.5px; margin:0 0 8px; }
.scroll { overflow-x:auto; -webkit-overflow-scrolling:touch; margin:8px 0 14px; border:1px solid var(--border); border-radius:8px; }
table { border-collapse:collapse; width:100%; min-width:760px; font-size:13.5px; background:var(--panel); }
th, td { border-bottom:1px solid var(--border); border-right:1px solid var(--border); padding:6px 10px; text-align:left; vertical-align:top; }
td.n { text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
th { background:var(--panel2); color:var(--muted); font-weight:600; white-space:nowrap; }
tr:last-child td { border-bottom:none; }
code { font-family:"JetBrains Mono",ui-monospace,Menlo,monospace; background:var(--code); padding:1px 5px; border-radius:5px; font-size:12px; overflow-wrap:anywhere; }
pre { background:var(--code); border:1px solid var(--border); border-radius:8px; padding:10px 12px; overflow-x:auto; }
pre code { padding:0; background:none; }
.box { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:12px 16px; margin:12px 0; }
.mu { color:var(--muted); }
.nav { font-size:13.5px; margin:0 0 14px; }
"""


# ── パスの正規化（外に出さない） ──

def safe_resolve(base: Path, rel: str) -> Path | None:
    """``base`` の下の実在するファイルなら解決後のパス、外に出る・無いなら None。``rel`` は URL の復号済みの相対パス。"""
    if not rel or "\x00" in rel or "\\" in rel or rel.startswith("/") or os.path.isabs(rel):
        return None
    if any(part in ("..", "") for part in rel.split("/")):
        return None
    try:
        base_r = base.resolve(strict=True)
        target = (base_r / rel).resolve(strict=True)  # シンボリックリンクはたどってから確かめる
    except (OSError, RuntimeError):
        return None
    if not target.is_relative_to(base_r) or not target.is_file():
        return None
    return target


def allowed_html(name: str) -> bool:
    return name in HTML_FIXED or bool(HTML_PATTERN.match(name))


# ── Markdown → HTML ──

def md_to_html(text: str) -> str:
    import markdown
    from markdown.extensions.toc import slugify_unicode

    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc"],
        extension_configs={"toc": {"slugify": slugify_unicode}},
    )
    return body.replace("<table>", '<div class="scroll"><table>').replace("</table>", "</table></div>")


def page(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="ja"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f"<body><main>{body}</main></body></html>"
    )


def first_heading(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


# ── 目次 ──

def list_rc(council_dir: Path) -> list[dict]:
    """rc/*.md を議題の番号（見出しの RC-n。無ければファイル名）順 → 段（R0〜R3）順 → 日付順に並べる。"""
    out = []
    rc_dir = council_dir / "rc"
    if not rc_dir.is_dir():
        return out
    for p in rc_dir.glob("*.md"):
        if safe_resolve(council_dir, f"rc/{p.name}") is None:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue  # 途中で消えた・読めない 1 件は飛ばす
        title = first_heading(text) or p.stem
        m = RC_NAME.match(p.name)
        d = DATE_RE.search(title)
        rc = RC_NO.search(title)  # 議題の番号は見出しの RC-n を正とする（ファイル名の番号とずれることがある）
        agenda = int(rc.group(1)) if rc else (int(m.group(2)) if m else 999)
        out.append({
            "name": p.name,
            "stage": int(m.group(1)) if m else 99,
            "agenda": agenda,
            "date": d.group(0) if d else "",
            "title": title,
        })
    out.sort(key=lambda r: (r["agenda"], r["stage"], r["date"], r["name"]))
    return out


def decision_headings(text: str) -> list[tuple[str, str]]:
    """decisions.md の ``## `` 見出しと、toc の拡張と同じアンカー。"""
    from markdown.extensions.toc import slugify_unicode

    out, seen = [], set()
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line.startswith("## "):
            continue
        title = line[3:].strip()
        slug = slugify_unicode(title, "-")
        base, n = slug, 1
        while slug in seen:  # toc の拡張と同じく重複には _1, _2 を付ける
            slug = f"{base}_{n}"
            n += 1
        seen.add(slug)
        out.append((title, slug))
    return out


def scoreboard_table(text: str) -> str:
    """scoreboard.md の表の部分だけを HTML にする。"""
    lines = [l for l in text.splitlines() if l.lstrip().startswith("|")]
    return md_to_html("\n".join(lines)) if lines else '<p class="mu">表がまだありません。</p>'


def index_html(root: Path) -> str:
    council = root / COUNCIL_REL
    esc = html.escape
    parts = ['<h1>絶対回収率プラスにするマン 会議の記録</h1>',
             f'<p class="sub">要求のたびに <code>{esc(str(COUNCIL_REL))}/</code> を読み直す（常に最新）。読み取り専用。</p>',
             f'<p class="nav"><a href="{COUNCIL_PREFIX}protocol.md">会議の進め方（protocol.md）</a> · '
             f'<a href="{COUNCIL_PREFIX}decisions.md">決定（decisions.md）</a> · '
             f'<a href="{COUNCIL_PREFIX}scoreboard.md">予想の成績表（scoreboard.md）</a></p>']

    parts.append("<h2>議題（rc）</h2>")
    rows = list_rc(council)
    if rows:
        parts.append('<div class="scroll"><table><thead><tr><th>議題</th><th>段</th><th>日付</th><th>資料</th></tr></thead><tbody>')
        for r in rows:
            agenda = f"RC-{r['agenda']}" if r["agenda"] != 999 else "－"
            stage = f"R{r['stage']}" if r["stage"] != 99 else "－"
            href = COUNCIL_PREFIX + "rc/" + urllib.parse.quote(r["name"])
            parts.append(f"<tr><td>{agenda}</td><td>{stage}</td><td>{esc(r['date'])}</td>"
                         f'<td><a href="{href}">{esc(r["title"])}</a></td></tr>')
        parts.append("</tbody></table></div>")
    else:
        parts.append('<p class="mu">rc/ に資料がありません。</p>')

    others = sorted(p.relative_to(council).as_posix() for p in council.rglob("*.md")
                    if p.parent != council / "rc" and p.parent != council) if council.is_dir() else []
    top_extra = sorted(p.name for p in council.glob("*.md")
                       if p.name not in {"protocol.md", "decisions.md", "scoreboard.md"}) if council.is_dir() else []
    extra = [x for x in top_extra + others if safe_resolve(council, x) is not None]
    if extra:
        parts.append("<h2>ほかの資料</h2><ul>")
        for rel in extra:
            parts.append(f'<li><a href="{COUNCIL_PREFIX}{urllib.parse.quote(rel)}">{esc(rel)}</a></li>')
        parts.append("</ul>")

    parts.append("<h2>決定（decisions.md）</h2>")
    dec = safe_resolve(council, "decisions.md")
    if dec:
        parts.append("<ul>")
        for title, slug in decision_headings(dec.read_text(encoding="utf-8", errors="replace")):
            parts.append(f'<li><a href="{COUNCIL_PREFIX}decisions.md#{urllib.parse.quote(slug)}">{esc(title)}</a></li>')
        parts.append("</ul>")
    else:
        parts.append('<p class="mu">decisions.md がありません。</p>')

    parts.append("<h2>予想の成績表（scoreboard.md）</h2>")
    sb = safe_resolve(council, "scoreboard.md")
    parts.append(scoreboard_table(sb.read_text(encoding="utf-8", errors="replace")) if sb else '<p class="mu">scoreboard.md がありません。</p>')

    parts.append("<h2>関係の HTML</h2>")
    hdir = root / HTML_REL
    names = sorted(p.name for p in hdir.glob("*.html") if allowed_html(p.name)) if hdir.is_dir() else []
    names = [n for n in names if safe_resolve(hdir, n) is not None]
    if names:
        parts.append("<ul>" + "".join(f'<li><a href="{HTML_PREFIX}{urllib.parse.quote(n)}">{esc(n)}</a></li>' for n in names) + "</ul>")
    else:
        parts.append('<p class="mu">関係の HTML がありません。</p>')
    return page("会議の記録", "".join(parts))


def md_page(root: Path, rel: str) -> str | None:
    target = safe_resolve(root / COUNCIL_REL, rel)
    if target is None or target.suffix != ".md":
        return None
    text = target.read_text(encoding="utf-8", errors="replace")
    nav = f'<p class="nav"><a href="/">目次へ</a> · <code>{html.escape(rel)}</code></p>'
    return page(first_heading(text) or rel, nav + md_to_html(text))


def html_file(root: Path, name: str) -> bytes | None:
    if "/" in name or not allowed_html(name):
        return None
    target = safe_resolve(root / HTML_REL, name)
    return target.read_bytes() if target else None


# ── サーバ ──

CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src 'self' data:; base-uri 'none'; form-action 'none'"
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})


def host_allowed(host_header: str) -> bool:
    """Host が手元の名前（127.0.0.1・localhost、番号は問わない）のときだけ答える。"""
    host = host_header.strip().lower()
    if host.startswith("["):
        name = host.split("]", 1)[0] + "]"
    else:
        name = host.split(":", 1)[0]
    return name in LOCAL_HOSTS or (os.environ.get("KEIBA_COUNCIL_ALLOW_ANY_HOST") == "1")

def make_handler(root: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "council-viewer"

        def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8", csp: bool = False) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            if csp:  # Markdown の生の HTML にスクリプトが混ざっても動かさない
                self.send_header("Content-Security-Policy", CSP)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _not_found(self) -> None:
            self._send(404, page("見つかりません", '<h1>見つかりません</h1><p><a href="/">目次へ</a></p>').encode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            if not host_allowed(self.headers.get("Host", "")):  # DNS rebinding で手元のブラウザから読まれないように
                return self._send(403, b"forbidden", "text/plain; charset=utf-8")
            raw = urllib.parse.urlsplit(self.path).path
            path = urllib.parse.unquote(raw)
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, index_html(root).encode("utf-8"), csp=True)
                if path == "/healthz":
                    import markdown  # noqa: F401  markdown が無い python で動いていたら 500 にして ensure に起動し直させる
                    return self._send(200, b"ok", "text/plain; charset=utf-8")
                if path.startswith(COUNCIL_PREFIX):
                    body = md_page(root, path[len(COUNCIL_PREFIX):])
                    return self._send(200, body.encode("utf-8"), csp=True) if body else self._not_found()
                if path.startswith(HTML_PREFIX):
                    data = html_file(root, path[len(HTML_PREFIX):])
                    return self._send(200, data) if data is not None else self._not_found()
            except Exception as e:  # noqa: BLE001  1 件の失敗でサーバを止めない
                self.log_error("error: %r", e)
                return self._send(500, page("エラー", "<h1>読み込みに失敗しました</h1>").encode("utf-8"))
            return self._not_found()

        do_HEAD = do_GET

    return Handler


def main(argv: list[str] | None = None) -> int:
    from src.config.ports import council_port

    ap = argparse.ArgumentParser(description="会議の記録の読み取り専用ビューア")
    ap.add_argument("--host", default=os.environ.get("KEIBA_COUNCIL_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=None, help="既定は KEIBA_PORT_COUNCIL（config/ports.env）")
    ap.add_argument("--root", default=os.environ.get("KEIBA_COUNCIL_ROOT", str(DEFAULT_ROOT)),
                    help="docs/ を読むリポジトリの根（既定はこのコードのリポジトリ）")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    port = a.port if a.port is not None else council_port()
    srv = ThreadingHTTPServer((a.host, port), make_handler(root))
    # Cursor の Remote-SSH はこの行の URL を拾って転送の候補にする
    print(f"Listening on http://{a.host}:{port}/ (root={root})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
