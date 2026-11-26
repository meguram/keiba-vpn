"""「絶対回収率プラスにするマン」の会議の記録を、ブラウザで読むための小さな読み取り専用サーバ。

    python -m src.scripts.docs.council_viewer                       # 127.0.0.1:KEIBA_PORT_COUNCIL（config/ports.env）
    python -m src.scripts.docs.council_viewer --port 8090 --root /path/to/keiba-vpn

- 要求のたびに ``docs/operations/roi_plus_council/`` の Markdown を読み、HTML にして返す（キャッシュしない。常に最新）。
- 加えて ``docs/html/operations/final/`` と ``docs/html/operations/`` の直下の HTML（名前は英数字・``_``・``-`` だけ）をそのまま返す。
- 読むのは上の 3 か所だけ。サブフォルダの HTML は出さない。``..``・絶対パス・シンボリックリンクで外に出る要求は 404。書き込みはしない。
- 認証が無いので、既定は 127.0.0.1 で待ち受ける（Cursor の Remote-SSH・ssh -L で手元へ転送して見る）。
- ``/live`` は会議のライブ（吹き出し）。画面は ``/live/api/sessions``・``/live/api/events`` を 3 秒おきに取りに来る
  （SSE は Cursor のポート転送でためられて届かないことがあったので、2026-11-26 に取りに行く形へ替えた）。中身は council_live.py。
- GCS・外部の網は読まない（費用 0 円）。
"""
from __future__ import annotations

import argparse
import html
import os
import posixpath
import re
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
COUNCIL_REL = Path("docs/operations/roi_plus_council")
HTML_REL = Path("docs/html/operations/final")
# URL はファイルの相対パスをそのまま映す（Markdown の相対リンクがそのまま効く）
COUNCIL_PREFIX = "/council/"
DOC_ALIAS = "/doc/"  # /council/ と同じ
EASY_SUMMARY = "easy_summary.md"
HTML_PREFIX = "/html/operations/final/"
HTML_OPS_REL = Path("docs/html/operations")
HTML_OPS_PREFIX = "/html/operations/"
# URL の接頭辞 → 置き場。長い接頭辞（final）を先に照らす
HTML_DIRS = ((HTML_PREFIX, HTML_REL), (HTML_OPS_PREFIX, HTML_OPS_REL))
HTML_NAME = re.compile(r"^[A-Za-z0-9_\-]+\.html$")
# 目次の「関係の HTML」に出す、会議に関係する HTML
HTML_FIXED = frozenset({"longshot_course_rules.html", "megu_cycles_50.html", "megu_why_below_100.html"})
HTML_PATTERN = re.compile(r"^rules_council_[A-Za-z0-9_\-]+\.html$")
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
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
    """直下の *.html なら出す（名前の形だけで判定。サブフォルダ・``..`` は形に合わない）。"""
    return bool(HTML_NAME.fullmatch(name))  # match の $ は末尾の改行の前でも合うので fullmatch


def council_html(name: str) -> bool:
    """目次の「関係の HTML」に出す名前か。"""
    return name in HTML_FIXED or bool(HTML_PATTERN.match(name))


def html_title(path: Path) -> str:
    """HTML の <title>（先頭 64 KiB から）。無ければ空。"""
    try:
        with path.open("rb") as f:
            head = f.read(65536).decode("utf-8", errors="replace")
    except OSError:
        return ""
    m = TITLE_RE.search(head)
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).split()) if m else ""


def list_html(root: Path, rel: Path) -> list[tuple[str, Path]]:
    """``rel`` の直下の出せる HTML の（URL に使う元の名前, 解決後のパス）（名前順）。"""
    hdir = root / rel
    if not hdir.is_dir():
        return []
    out = []
    for p in sorted(hdir.glob("*.html")):
        if allowed_html(p.name):
            t = safe_resolve(hdir, p.name)
            if t is not None:
                out.append((p.name, t))
    return out


# ── Markdown → HTML ──

HREF_RE = re.compile(r'(<a\s[^>]*?href=")([^"]*)(")')


def rewrite_links(body: str, base_dir: str) -> str:
    """Markdown の中の相対リンクを、council の根からの URL（/council/...）に直す。

    目次（/）に本文を埋め込んだときも、easy/RC-02.md のようなリンクがそのまま開ける。外（.. で根の上）・絶対・スキームは触らない。
    """
    def fix(m: re.Match) -> str:
        href = m.group(2)
        if not href or href.startswith(("#", "/", "mailto:")) or re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*:", href):
            return m.group(0)
        path, sep, frag = href.partition("#")
        joined = posixpath.normpath(posixpath.join(base_dir, urllib.parse.unquote(path))) if path else ""
        if not joined or joined == "." or joined.startswith(".."):
            return m.group(0)
        return m.group(1) + COUNCIL_PREFIX + urllib.parse.quote(joined) + (sep + frag if sep else "") + m.group(3)

    return HREF_RE.sub(fix, body)


MD_EXTENSIONS = ["tables", "fenced_code"]  # 文書（/council）とライブで同じ描き方にする
A_TAG_RE = re.compile(r'<a href="([^"]*)"')
IMG_TAG_RE = re.compile(r"<img\b[^>]*>")
ALT_RE = re.compile(r'\balt="([^"]*)"')
SAFE_SCHEMES = ("http", "https", "mailto")
_safe_md_lock = threading.Lock()
_safe_md = None


def _wrap_tables(body: str) -> str:
    return body.replace("<table>", '<div class="scroll"><table>').replace("</table>", "</table></div>")


def md_to_html(text: str, base_dir: str | None = None) -> str:
    """``base_dir`` は council の根からの、その Markdown のあるフォルダ（根は ""）。None なら相対リンクを直さない。"""
    import markdown
    from markdown.extensions.toc import slugify_unicode

    body = markdown.markdown(
        text,
        extensions=MD_EXTENSIONS + ["toc"],
        extension_configs={"toc": {"slugify": slugify_unicode}},
    )
    if base_dir is not None:
        body = rewrite_links(body, base_dir)
    return _wrap_tables(body)


def safe_href(href: str) -> bool:
    """リンク先が http(s)・mailto・相対・ページ内なら True（javascript: などは通さない。実体参照・空白で隠しても見抜く）。"""
    raw = "".join(ch for ch in html.unescape(href) if ch > " " and ch != "\x7f")
    m = re.match(r"^([A-Za-z][A-Za-z0-9+.\-]*):", raw)
    return m is None or m.group(1).lower() in SAFE_SCHEMES


LIST_LINE_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _ordered(line: str) -> bool:
    return line.lstrip()[:1].isdigit()


def loosen_blocks(text: str) -> str:
    """会話の文は段落の直後に箇条書き・表・囲みのコードを続けて書くことが多い。Markdown が段落の続きと
    みなさないよう、その手前に空の行を入れる（囲みのコードの中は触らない）。"""
    out: list[str] = []
    fence, prev = False, ""
    lines = text.split("\n")
    # 2 字下げで入れ子を書いた文（字下げが 4 の倍数でない箇条がある）は、箇条の字下げを倍にする
    two = any(LIST_LINE_RE.match(l) and (len(l) - len(l.lstrip())) % 4 == 2 for l in lines)
    for line in lines:
        st = line.lstrip()
        opening = st.startswith(("```", "~~~"))
        if not fence and LIST_LINE_RE.match(line):
            ind = len(line) - len(st)
            if two and ind:
                line = " " * (ind * 2) + st
        if not fence and prev.strip() and line.strip():
            p_list, p_table = bool(LIST_LINE_RE.match(prev)), prev.lstrip().startswith("|")
            c_list, c_table = bool(LIST_LINE_RE.match(line)), st.startswith("|")
            if opening or (c_list and not p_list and not prev.startswith((" ", "\t"))) \
                    or (c_table and not p_table) or (p_table and not c_table) \
                    or (p_list and not c_list and not line.startswith((" ", "\t"))) \
                    or (p_list and c_list and _ordered(prev) != _ordered(line) and _indent(line) <= _indent(prev)):
                out.append("")
        if opening:
            fence = not fence
        out.append(line)
        prev = line
    return "\n".join(out)


def safe_md_to_html(text: str) -> str:
    """会話の本文（外から来うる文）を Markdown として描く。生の HTML は描かずに文字として出す（エスケープ）。

    - Markdown の生の HTML（ブロック・インライン）の処理を外す → ``<script>``・``onerror=`` は ``&lt;`` の文字になる。
    - リンクは http(s)・mailto・相対だけ残し、ほかは href を外す。画像は読み込まずに代わりの文字にする。
    - 改行はそのまま改行にする（会話は 1 行ずつ書くことが多い）。
    """
    global _safe_md
    import markdown

    with _safe_md_lock:  # Markdown の実体は複数のスレッドで同時に使えない
        if _safe_md is None:
            md = markdown.Markdown(extensions=MD_EXTENSIONS + ["nl2br", "sane_lists"])
            md.preprocessors.deregister("html_block")
            md.inlinePatterns.deregister("html")
            _safe_md = md
        body = _safe_md.reset().convert(loosen_blocks(text))
    body = A_TAG_RE.sub(lambda m: m.group(0) if safe_href(m.group(1)) else "<a", body)
    body = IMG_TAG_RE.sub(lambda m: "［画像: " + ((ALT_RE.search(m.group(0)) or [None, ""])[1]) + "］", body)
    body = body.replace("<a href=", '<a rel="noopener noreferrer" target="_blank" href=')
    return _wrap_tables(body)


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
    return md_to_html("\n".join(lines), "") if lines else '<p class="mu">表がまだありません。</p>'


def index_html(root: Path) -> str:
    council = root / COUNCIL_REL
    esc = html.escape
    parts = ['<h1>絶対回収率プラスにするマン 会議の記録</h1>',
             f'<p class="sub">要求のたびに <code>{esc(str(COUNCIL_REL))}/</code> を読み直す（常に最新）。読み取り専用。</p>',
             f'<p class="nav"><a href="{COUNCIL_PREFIX}protocol.md">会議の進め方（protocol.md）</a> · '
             f'<a href="{COUNCIL_PREFIX}decisions.md">決定（decisions.md）</a> · '
             f'<a href="{COUNCIL_PREFIX}scoreboard.md">予想の成績表（scoreboard.md）</a> · '
             '<a href="/live"><strong>ライブ（いまの議論）</strong></a></p>']

    easy = safe_resolve(council, EASY_SUMMARY)
    if easy:  # 小学生でもわかる説明を、リンクではなく本文のまま一番上に出す
        parts.append(f'<div class="box">{md_to_html(easy.read_text(encoding="utf-8", errors="replace"), "")}</div>')

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
                       if p.name not in {"protocol.md", "decisions.md", "scoreboard.md", EASY_SUMMARY}) if council.is_dir() else []
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
    names = sorted(p.name for p in hdir.glob("*.html") if council_html(p.name)) if hdir.is_dir() else []
    names = [n for n in names if safe_resolve(hdir, n) is not None]
    if names:
        parts.append("<ul>" + "".join(f'<li><a href="{HTML_PREFIX}{urllib.parse.quote(n)}">{esc(n)}</a></li>' for n in names) + "</ul>")
    else:
        parts.append('<p class="mu">関係の HTML がありません。</p>')

    parts.append("<h2>成果の HTML</h2>")
    any_html = False
    for prefix, rel in HTML_DIRS:
        files = list_html(root, rel)
        if not files:
            continue
        any_html = True
        parts.append(f"<h3><code>{esc(rel.as_posix())}/</code></h3>")
        parts.append('<div class="scroll"><table><thead><tr><th>ファイル</th><th>題名</th></tr></thead><tbody>')
        for name, p in files:
            href = prefix + urllib.parse.quote(name)
            parts.append(f'<tr><td><a href="{href}">{esc(name)}</a></td><td>{esc(html_title(p))}</td></tr>')
        parts.append("</tbody></table></div>")
    if not any_html:
        parts.append('<p class="mu">成果の HTML がありません。</p>')
    return page("会議の記録", "".join(parts))


def md_page(root: Path, rel: str) -> str | None:
    target = safe_resolve(root / COUNCIL_REL, rel)
    if target is None or target.suffix != ".md":
        return None
    text = target.read_text(encoding="utf-8", errors="replace")
    nav = f'<p class="nav"><a href="/">目次へ</a> · <code>{html.escape(rel)}</code></p>'
    return page(first_heading(text) or rel, nav + md_to_html(text, posixpath.dirname(rel)))


def html_file(root: Path, name: str, rel: Path = HTML_REL) -> bytes | None:
    """``rel``（既定は final）の直下の HTML の中身。名前の形に合わない・無いなら None。"""
    if "/" in name or not allowed_html(name):
        return None
    target = safe_resolve(root / rel, name)
    return target.read_bytes() if target else None


# ── サーバ ──

# 成果の HTML は自前の生成物だが <script> を含むものがある。sandbox で origin を分け、/council/・/live/api を読ませない
HTML_CSP = "sandbox allow-scripts allow-popups allow-popups-to-escape-sandbox"
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

# スクリプトは応答ごとの nonce の付いた自前の 1 本だけ動かす（本文に onerror= などが紛れても動かない）
LIVE_CSP = ("default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-{nonce}'; connect-src 'self'; "
            "base-uri 'none'; form-action 'none'")
LIVE_POLL_MS = 3000  # 画面が取りに行く間隔（手元の 127.0.0.1 だけ。費用 0 円）
LIVE_LIMITS = (20, 30, 50, 100)
LIVE_MAX_LIMIT = 200
LIVE_CSS = """
:root { --s1:#c0262d; --s2:#1d6fb8; --s3:#1d7f45; --p1:#a04a00; --p2:#8a3fa0; --sc:#5a46b8; --so:#8a6100; --sx:#5b6578; --sn:#5b6578; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --s1:#ff7b7b; --s2:#7aa9ff; --s3:#4ade80; --p1:#fdba74; --p2:#e2a6f5; --sc:#b9a6ff; --so:#fbbf24; --sx:#9aa7c7; --sn:#9aa7c7; } }
:root[data-theme="dark"] { --s1:#ff7b7b; --s2:#7aa9ff; --s3:#4ade80; --p1:#fdba74; --p2:#e2a6f5; --sc:#b9a6ff; --so:#fbbf24; --sx:#9aa7c7; --sn:#9aa7c7; }
main { max-width:980px; padding-top:0; }
.bar { position:sticky; top:0; background:var(--bg); padding:10px 0 8px; border-bottom:1px solid var(--border); z-index:3;
  font-size:13.5px; display:flex; flex-wrap:wrap; gap:6px 12px; align-items:center; }
.bar strong { font-size:15px; }
.bar select { font:inherit; color:var(--text); background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:3px 6px; max-width:100%; }
.bar label { white-space:nowrap; }
#sess { min-width:0; max-width:min(420px, 100%); }
#log { padding:10px 0 72px; }
.row { display:flex; margin:10px 0; }
.row.l { justify-content:flex-start; } .row.r { justify-content:flex-end; } .row.c { justify-content:center; }
.bub { max-width:min(78%, 760px); min-width:0; background:color-mix(in srgb, var(--c) 9%, var(--panel)); border:1px solid color-mix(in srgb, var(--c) 35%, var(--border));
  border-radius:14px; padding:8px 12px 9px; position:relative; }
.row.l .bub { border-top-left-radius:4px; } .row.r .bub { border-top-right-radius:4px; }
.bub.s1 { border-width:2px; }
.bub.dec { border:2px solid var(--s1); box-shadow:0 0 0 3px color-mix(in srgb, var(--s1) 18%, transparent); }
.hd { font-size:12px; color:var(--muted); display:flex; gap:6px; flex-wrap:wrap; align-items:center; margin-bottom:2px; }
.who { color:var(--c); font-weight:700; font-size:13px; }
.tag { font-size:11px; line-height:1.6; padding:0 7px; border-radius:9px; border:1px solid currentColor; white-space:nowrap; }
.tag.dec { background:var(--s1); color:#fff; border-color:var(--s1); font-weight:700; }
.md { overflow-wrap:anywhere; font-size:14.5px; line-height:1.65; }
.md > :first-child { margin-top:0; } .md > :last-child { margin-bottom:0; }
.md p { margin:.35em 0; } .md ul, .md ol { margin:.35em 0; padding-left:1.4em; }
.md h1, .md h2, .md h3, .md h4 { font-size:15px; margin:.7em 0 .3em; padding:0; border:0; }
.md blockquote { margin:.4em 0; padding:.1em .8em; border-left:3px solid var(--border); color:var(--muted); }
.md pre { font-size:12px; margin:.4em 0; } .md .scroll { margin:.4em 0; display:inline-block; max-width:100%; vertical-align:top; } .md table { min-width:0; width:auto; font-size:12.5px; }
.md th, .md td { padding:3px 8px; }
.clip { max-height:15em; overflow:hidden; -webkit-mask-image:linear-gradient(#000 70%, transparent); mask-image:linear-gradient(#000 70%, transparent); }
.more { margin-top:4px; font:inherit; font-size:12.5px; color:var(--accent); background:none; border:0; padding:0; cursor:pointer; }
.mini { font-size:12px; color:var(--muted); max-width:min(78%, 760px); padding:2px 10px; border-radius:10px; overflow-wrap:anywhere; }
.mini.call { color:var(--c); border:1px dashed color-mix(in srgb, var(--c) 45%, var(--border)); }
.mini.notice { text-align:center; }
.tools { font-size:11.5px; color:var(--muted); background:none; border:1px solid var(--border); border-radius:10px; padding:1px 9px; cursor:pointer; font-family:inherit; }
.tlist { font-size:11.5px; color:var(--muted); margin:2px 0 0; padding-left:1.2em; max-width:min(78%, 760px); }
.grp { display:flex; flex-direction:column; }
.row.l .grp { align-items:flex-start; } .row.r .grp { align-items:flex-end; }
#news { position:fixed; left:50%; transform:translateX(-50%); bottom:18px; padding:7px 16px; border-radius:18px; border:1px solid var(--accent);
  background:var(--panel); color:var(--accent); cursor:pointer; font:inherit; font-size:13.5px; box-shadow:0 2px 10px rgba(0,0,0,.18); z-index:4; }
.empty { color:var(--muted); text-align:center; padding:40px 0; }
.s1 { --c:var(--s1); } .s2 { --c:var(--s2); } .s3 { --c:var(--s3); } .p1 { --c:var(--p1); } .p2 { --c:var(--p2); }
.sc { --c:var(--sc); } .so { --c:var(--so); } .sx { --c:var(--sx); } .sn { --c:var(--sn); }
@media (max-width:600px) { .bub, .mini, .tlist { max-width:94%; } .bar { font-size:13px; } main { padding-left:10px; padding-right:10px; } }
"""
LIVE_JS = r"""
(function () {
  var POLL = %POLL%, LEFT = {s1:1, s2:1, s3:1, p1:1, p2:1, sx:1};
  var $ = function (id) { return document.getElementById(id); };
  var log = $('log'), st = $('st'), news = $('news'), selS = $('sess'), selN = $('lim'), chkT = $('tools');
  var sid = '', seq = 0, epoch = '', unread = 0, timer = 0, busy = false, lastTool = null, gen = 0, lastSig = null;
  function load(k, d) { try { return localStorage.getItem('live.' + k) || d; } catch (e) { return d; } }
  function save(k, v) { try { localStorage.setItem('live.' + k, v); } catch (e) {} }
  selN.value = load('lim', '30'); if (!selN.value) selN.value = '30';
  chkT.checked = load('tools', '0') === '1';
  function atBottom() { return window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 80; }
  function toBottom() { window.scrollTo(0, document.documentElement.scrollHeight); }
  function el(tag, cls, text) { var x = document.createElement(tag); if (cls) x.className = cls; if (text != null) x.textContent = text; return x; }
  function hhmm(ts) { var d = new Date(ts); return isNaN(d) ? '' : d.toLocaleString('ja-JP', {month:'numeric', day:'numeric', hour:'2-digit', minute:'2-digit'}); }
  function side(e) { return e.kind === 'notice' ? 'c' : (LEFT[e.cls] ? 'l' : 'r'); }
  function showNews() { news.hidden = unread === 0; news.textContent = '新しい発言 ' + unread + ' 件 ↓'; }
  window.addEventListener('scroll', function () { if (atBottom()) { if (unread) trim(); unread = 0; showNews(); } }, {passive: true});
  news.addEventListener('click', function () { unread = 0; showNews(); toBottom(); });
  function bubble(e) {
    var row = el('div', 'row ' + side(e));
    if (e.kind === 'notice' || e.kind === 'call') {
      row.appendChild(el('div', 'mini ' + e.kind + ' ' + e.cls, hhmm(e.ts) + ' ' + (e.kind === 'call' ? e.speaker + ' ' : '') + e.text));
      return row;
    }
    var b = el('div', 'bub ' + e.cls + (e.decision ? ' dec' : '')), h = el('div', 'hd');
    h.appendChild(el('span', 'who', e.speaker));
    h.appendChild(el('span', '', hhmm(e.ts)));
    if (e.topic) h.appendChild(el('span', 'tag', e.topic));
    if (e.stage) h.appendChild(el('span', 'tag', e.stage));
    if (e.decision) h.appendChild(el('span', 'tag dec', '決定'));
    b.appendChild(h);
    var body = el('div', 'md'); body.innerHTML = e.html || '';  /* サーバで生の HTML を外して描いたもの */
    b.appendChild(body); row.appendChild(b);
    return row;
  }
  function clampIfLong(row) {
    var body = row.querySelector('.md'); if (!body || body.scrollHeight < 300) return;
    body.classList.add('clip');
    var btn = el('button', 'more', '続きを開く');
    btn.addEventListener('click', function () { var open = body.classList.toggle('clip'); btn.textContent = open ? '続きを開く' : 'たたむ'; });
    body.parentNode.appendChild(btn);
  }
  function toolGroup(e) {
    var row = el('div', 'row ' + side(e)), g = el('div', 'grp'), btn = el('button', 'tools'), ul = el('ul', 'tlist');
    ul.hidden = true; g.appendChild(btn); g.appendChild(ul); row.appendChild(g);
    row._n = 0; row._names = {}; row._who = e.speaker;
    btn.addEventListener('click', function () { ul.hidden = !ul.hidden; });
    row._add = function (x) {
      row._n += 1; var name = x.text.split(':')[0]; row._names[name] = (row._names[name] || 0) + 1;
      var parts = []; for (var k in row._names) parts.push(k + ' ' + row._names[k]);
      btn.textContent = row._who + ' のツール ' + row._n + ' 件（' + parts.join('・') + '）';
      ul.appendChild(el('li', '', hhmm(x.ts) + ' ' + x.text));
    };
    return row;
  }
  function append(e) {
    if (e.kind === 'tool') {
      if (!chkT.checked) return;
      if (!lastTool || lastTool._who !== e.speaker || lastTool !== log.lastChild) { lastTool = toolGroup(e); log.appendChild(lastTool); }
      lastTool._add(e); return;
    }
    var row = bubble(e); log.appendChild(row); clampIfLong(row);
  }
  function trim() {
    var lim = Math.max(1, +selN.value || 30), rows = log.querySelectorAll('.row'), talk = 0;
    for (var i = rows.length - 1; i >= 0; i--) {
      if (rows[i]._add) continue;  /* ツールのまとめは数えない */
      talk += 1;
      if (talk > lim) { while (log.firstChild && log.firstChild !== rows[i].nextSibling) log.removeChild(log.firstChild); break; }
    }
  }
  function status(t) { st.textContent = t; }
  function get(url, ok, ng) {
    var x = new XMLHttpRequest(); x.open('GET', url); x.timeout = 15000;
    x.onload = function () { if (x.status === 200) { try { ok(JSON.parse(x.responseText)); } catch (err) { ng(); } } else ng(); };
    x.onerror = x.ontimeout = ng; x.send();
  }
  function schedule(ms) { clearTimeout(timer); timer = setTimeout(poll, ms); }
  function reset() { gen += 1; seq = 0; epoch = ''; unread = 0; showNews(); lastTool = null; log.textContent = ''; }
  function poll() {
    if (!sid || busy) { schedule(POLL); return; }
    busy = true;
    var g = gen, url = '/live/api/events?session=' + encodeURIComponent(sid) + '&since=' + seq + '&limit=' + selN.value;
    get(url, function (d) {
      busy = false;
      if (g !== gen) { schedule(0); return; }  /* 待つ間に会議・件数を替えた */
      if (d.epoch !== epoch) {  /* サーバを起動し直した・読み直した。番号が替わるので初めから取り直す */
        var had = epoch; epoch = d.epoch;
        if (had) { reset(); schedule(0); return; }
      }
      var follow = atBottom() || seq === 0, talk = 0, first = seq === 0;
      (d.events || []).forEach(function (e) { append(e); if (e.kind !== 'tool') talk += 1; seq = Math.max(seq, e.seq); });
      if (first && !log.firstChild) log.appendChild(el('div', 'empty', 'まだ発言がありません。'));
      else if (talk && log.querySelector('.empty')) log.removeChild(log.querySelector('.empty'));
      if (follow) { trim(); toBottom(); } else if (talk) { unread += talk; showNews(); }
      status('更新 ' + new Date().toLocaleTimeString('ja-JP') + '（' + POLL / 1000 + ' 秒おき）');
      schedule(document.hidden ? POLL * 5 : POLL);
    }, function () { busy = false; status('つながりません。再試行しています…'); schedule(g !== gen ? 0 : POLL * 2); });
  }
  function label(s) {
    var mem = (s.members || []).map(function (m) { return m.replace('roi-plus-man-', '').replace('pedigree-', '血統'); }).join('・');
    return s.name + '（' + hhmm(s.mtime * 1000) + (mem ? '・' + mem : '') + '）';
  }
  function sessions(first) {
    get('/live/api/sessions', function (d) {
      var q = (location.search.match(/[?&]s=([0-9a-f-]+)/) || [])[1];
      var list = d.sessions || [], keep = sid || q || load('sid', ''), found = false;
      var sig = list.map(function (s) { return s.id + label(s); }).join('|');
      list.forEach(function (s) { if (s.id === keep) found = true; });
      if (sig !== lastSig) {  /* 変わったときだけ作り直す（開いている選択肢を閉じない） */
        lastSig = sig; selS.textContent = '';
        list.forEach(function (s) { var o = el('option', '', label(s)); o.value = s.id; selS.appendChild(o); });
        if (!list.length) { var o = el('option', '', '会議のセッションが見つかりません'); o.value = ''; selS.appendChild(o); }
      }
      var want = found ? keep : (list[0] ? list[0].id : '');
      selS.value = want;
      if (want !== sid) { sid = want; reset(); schedule(0); }
      if (!list.length) { log.textContent = ''; log.appendChild(el('div', 'empty', '最近の会議のセッションがありません（' + d.hours + ' 時間以内）。')); }
    }, function () { status('セッションの一覧を取れません。再試行しています…'); });
  }
  selS.addEventListener('change', function () { sid = selS.value; save('sid', sid); reset(); schedule(0); });
  selN.addEventListener('change', function () { save('lim', selN.value); reset(); schedule(0); });
  chkT.addEventListener('change', function () { save('tools', chkT.checked ? '1' : '0'); reset(); schedule(0); });
  document.addEventListener('visibilitychange', function () { if (!document.hidden) schedule(0); });
  sessions(true); setInterval(sessions, 30000);
})();
"""


def live_page(nonce: str = "") -> str:
    opts = "".join(f'<option value="{n}">{n} 件</option>' for n in LIVE_LIMITS)
    body = (
        '<div class="bar"><strong>会議のライブ</strong>'
        '<select id="sess" aria-label="どの担当の会議か"><option>読み込み中…</option></select>'
        f'<label>直近 <select id="lim" aria-label="表示する件数">{opts}</select></label>'
        '<label><input type="checkbox" id="tools"> ツールの行</label>'
        '<a href="/">目次へ</a><span id="st" class="mu">読み込み中…</span></div>'
        '<div id="log"></div><button id="news" hidden></button>'
        f"<style>{LIVE_CSS}</style><script nonce=\"{html.escape(nonce)}\">{LIVE_JS.replace('%POLL%', str(LIVE_POLL_MS))}</script>"
    )
    return page("会議のライブ", body)


EVENT_FIELDS = ("seq", "ts", "speaker", "cls", "kind", "stage", "topic", "decision")


def event_json(e: dict) -> dict:
    """画面に渡す形。本文（text）は Markdown を安全に描いた html にして渡す（描いた結果は出来事に覚えておく）。"""
    out = {k: e.get(k) for k in EVENT_FIELDS}
    if e.get("kind") == "text":
        h = e.get("html")
        if h is None:
            h = e["html"] = safe_md_to_html(e.get("text", ""))
        out["html"] = h
    else:
        out["text"] = e.get("text", "")
    return out


def make_handler(root: Path, hub=None):
    if hub is None:
        from src.scripts.docs.council_live import default_hub

        hub = default_hub(root)

    class Handler(BaseHTTPRequestHandler):
        server_version = "council-viewer"

        def _json(self, obj: dict) -> None:
            import json as _json

            self._send(200, _json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _live_api(self, path: str) -> None:
            """画面が数秒おきに取りに来る（SSE は Cursor の転送でためられて届かないことがあるので、取りに行く形にした）。"""
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            if path == "/live/api/sessions":
                return self._json({"epoch": hub.epoch, "hours": round(hub.window_sec / 3600, 1), "sessions": hub.sessions()})
            sid = (q.get("session") or [""])[0]
            since = (q.get("since") or ["0"])[0]
            limit = (q.get("limit") or ["30"])[0]
            seq = int(since) if re.fullmatch(r"[0-9]{1,12}", since) else 0  # isdigit は ² なども通すので使わない
            lim = min(LIVE_MAX_LIMIT, max(1, int(limit))) if re.fullmatch(r"[0-9]{1,4}", limit) else 30
            feed = hub.feed(sid) if re.fullmatch(r"[0-9a-f\-]{8,64}", sid or "") else None
            if feed is None:
                return self._send(404, b'{"error":"no such council session"}', "application/json; charset=utf-8")
            evs = feed.since(seq, lim)
            return self._json({"epoch": f"{hub.epoch}-{feed.created:.3f}", "session": sid, "seq": feed.seq,
                               "events": [event_json(e) for e in evs]})

        def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8", csp: bool | str = False) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            if csp:  # Markdown の生の HTML にスクリプトが混ざっても動かさない（ライブの画面は自前のスクリプトだけ許す）
                self.send_header("Content-Security-Policy", csp if isinstance(csp, str) else CSP)
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
                if path == "/live":
                    import secrets

                    nonce = secrets.token_urlsafe(18)
                    return self._send(200, live_page(nonce).encode("utf-8"), csp=LIVE_CSP.format(nonce=nonce))
                if path in ("/live/api/sessions", "/live/api/events"):
                    return self._live_api(path)
                if path == "/healthz":
                    import markdown  # noqa: F401  markdown が無い python で動いていたら 500 にして ensure に起動し直させる
                    return self._send(200, b"ok", "text/plain; charset=utf-8")
                for prefix in (COUNCIL_PREFIX, DOC_ALIAS):
                    if path.startswith(prefix):
                        body = md_page(root, path[len(prefix):])
                        return self._send(200, body.encode("utf-8"), csp=True) if body else self._not_found()
                for prefix, rel in HTML_DIRS:
                    if path.startswith(prefix):
                        data = html_file(root, path[len(prefix):], rel)
                        return self._send(200, data, csp=HTML_CSP) if data is not None else self._not_found()
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
