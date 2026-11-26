"""「絶対回収率プラスにするマン」の会議の記録を、ブラウザで読むための小さな読み取り専用サーバ。

    python -m src.scripts.docs.council_viewer                       # 127.0.0.1:KEIBA_PORT_COUNCIL（config/ports.env）
    python -m src.scripts.docs.council_viewer --port 8090 --root /path/to/keiba-vpn

- 要求のたびに ``docs/operations/roi_plus_council/`` の Markdown を読み、HTML にして返す（キャッシュしない。常に最新）。
- 加えて ``docs/html/operations/final/`` と ``docs/html/operations/`` の直下の HTML（名前は英数字・``_``・``-`` だけ）をそのまま返す。
- 読むのは上の 3 か所だけ。サブフォルダの HTML は出さない。``..``・絶対パス・シンボリックリンクで外に出る要求は 404。書き込みはしない。
- 認証が無いので、既定は 127.0.0.1 で待ち受ける（Cursor の Remote-SSH・ssh -L で手元へ転送して見る）。
- GCS・外部の網は読まない（費用 0 円）。
"""
from __future__ import annotations

import argparse
import html
import os
import posixpath
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


def md_to_html(text: str, base_dir: str | None = None) -> str:
    """``base_dir`` は council の根からの、その Markdown のあるフォルダ（根は ""）。None なら相対リンクを直さない。"""
    import markdown
    from markdown.extensions.toc import slugify_unicode

    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc"],
        extension_configs={"toc": {"slugify": slugify_unicode}},
    )
    if base_dir is not None:
        body = rewrite_links(body, base_dir)
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

# 成果の HTML は自前の生成物だが <script> を含むものがある。sandbox で origin を分け、/council/・/live/stream を読ませない
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

LIVE_CSP = ("default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; "
            "base-uri 'none'; form-action 'none'")
LIVE_CSS = """
:root { --s1:#b4232c; --s2:#1d6fb8; --s3:#1d7f45; --sc:#6b4bc4; --so:#9a6200; --sx:#5b6578; --sn:#5b6578; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --s1:#f87171; --s2:#7aa9ff; --s3:#4ade80; --sc:#c4a7ff; --so:#fbbf24; --sx:#9aa7c7; --sn:#9aa7c7; } }
:root[data-theme="dark"] { --s1:#f87171; --s2:#7aa9ff; --s3:#4ade80; --sc:#c4a7ff; --so:#fbbf24; --sx:#9aa7c7; --sn:#9aa7c7; }
.bar { position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--border); z-index:2; font-size:13.5px; }
.ev { background:var(--panel); border:1px solid var(--border); border-left:4px solid var(--c, var(--border)); border-radius:8px; padding:8px 12px; margin:8px 0; }
.ev .h { font-size:12.5px; color:var(--muted); display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.ev .who { color:var(--c); font-weight:700; }
.ev .t { white-space:pre-wrap; overflow-wrap:anywhere; margin-top:4px; }
.ev.tool, .ev.notice, .ev.call { padding:3px 12px; font-size:12.5px; color:var(--muted); }
.ev.tool .t, .ev.notice .t, .ev.call .t { margin:0; display:inline; white-space:normal; }
.ev.call { color:var(--text); }
.tag { font-size:11px; padding:0 7px; border-radius:9px; border:1px solid currentColor; }
.dec { color:var(--bad); font-weight:700; }
body.hide-tools .ev.tool { display:none; }
#jump { position:fixed; right:16px; bottom:16px; padding:6px 12px; border-radius:16px; border:1px solid var(--border); background:var(--panel); color:var(--accent); cursor:pointer; }
.s1 { --c:var(--s1); } .s2 { --c:var(--s2); } .s3 { --c:var(--s3); } .sc { --c:var(--sc); } .so { --c:var(--so); } .sx { --c:var(--sx); } .sn { --c:var(--sn); }
"""
LIVE_JS = """
(function () {
  var log = document.getElementById('log'), st = document.getElementById('st'), jump = document.getElementById('jump');
  var follow = true, n = 0;
  function atBottom() { return window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 60; }
  window.addEventListener('scroll', function () { follow = atBottom(); jump.hidden = follow; });
  jump.addEventListener('click', function () { follow = true; jump.hidden = true; window.scrollTo(0, document.documentElement.scrollHeight); });
  document.getElementById('tools').addEventListener('change', function (e) { document.body.classList.toggle('hide-tools', !e.target.checked); });
  function el(tag, cls, text) { var x = document.createElement(tag); if (cls) x.className = cls; if (text) x.textContent = text; return x; }
  function hhmm(ts) { var d = new Date(ts); return isNaN(d) ? '' : d.toLocaleString('ja-JP', {month:'numeric', day:'numeric', hour:'2-digit', minute:'2-digit', second:'2-digit'}); }
  function add(ev) {
    var box = el('div', 'ev ' + ev.kind + ' ' + ev.cls), h = el('div', 'h');
    h.appendChild(el('span', '', hhmm(ev.ts)));
    h.appendChild(el('span', 'who', ev.speaker));
    if (ev.stage) h.appendChild(el('span', 'tag', ev.stage));
    if (ev.decision) h.appendChild(el('span', 'tag dec', '決定'));
    if (ev.kind === 'tool') h.appendChild(el('span', 'tag', 'ツール'));
    if (ev.kind === 'text') { box.appendChild(h); box.appendChild(el('div', 't', ev.text)); }
    else { h.appendChild(el('span', 't', ev.text)); box.appendChild(h); }
    log.appendChild(box); n += 1;
  }
  var es = new EventSource('/live/stream');
  es.onopen = function () { st.textContent = '接続中（' + n + ' 件）'; };
  es.onmessage = function (m) {
    add(JSON.parse(m.data)); st.textContent = '接続中（' + n + ' 件）';
    if (follow) window.scrollTo(0, document.documentElement.scrollHeight);
  };
  es.onerror = function () { st.textContent = '切れました。再接続を待っています（' + n + ' 件）'; };
})();
"""


def live_page() -> str:
    body = (
        '<div class="bar"><strong>会議のライブ</strong>（担当 rules-council と各号。数秒おきに記録の差分を流す） · '
        '<a href="/">目次へ</a> · <label><input type="checkbox" id="tools"> ツールの行を出す</label> · '
        '<span id="st" class="mu">接続しています…</span></div>'
        '<div id="log"></div><button id="jump" hidden>最新へ ↓</button>'
        f"<style>{LIVE_CSS}</style><script>{LIVE_JS}</script>"
    )
    return page("会議のライブ", body).replace("<body>", '<body class="hide-tools">', 1)


def make_handler(root: Path, feed=None):
    if feed is None:
        from src.scripts.docs.council_live import LiveFeed, default_project_dir, default_tmp_base

        live_dir = os.environ.get("KEIBA_COUNCIL_LIVE_DIR")
        feed = LiveFeed(Path(live_dir) if live_dir else default_project_dir(root), default_tmp_base())

    class Handler(BaseHTTPRequestHandler):
        server_version = "council-viewer"

        def _stream(self) -> None:
            """Server-Sent Events。読むのは feed の 1 本のスレッドだけで、ここは貯まった出来事を渡すだけ。"""
            import json as _json

            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            last = self.headers.get("Last-Event-ID") or (q.get("since") or ["0"])[0]
            seq = int(last) if str(last).isdigit() else 0
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            if self.command == "HEAD":
                return
            self.connection.settimeout(30)  # 応答しない相手で書き込みが止まり続けないように
            feed.attach()
            try:
                if feed.seq == 0:
                    feed.poll_once()
                if seq > feed.seq:  # 起動し直して番号が戻った。直近の分から流し直す
                    seq = 0
                pending = feed.since(seq)
                idle = 0.0
                while True:
                    for e in pending:
                        self.wfile.write(f"id: {e['seq']}\ndata: {_json.dumps(e, ensure_ascii=False)}\n\n".encode("utf-8"))
                        seq = e["seq"]
                    if pending:
                        self.wfile.flush()
                        idle = 0.0
                    pending = feed.wait(seq, 5.0)
                    if not pending:
                        idle += 5.0
                        if idle >= 15.0:  # 切れたことに気づくための空の行
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                            idle = 0.0
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                pass
            except Exception as e:  # noqa: BLE001  ヘッダを送った後なので 500 は書かずに閉じる
                self.log_error("stream error: %r", e)
            finally:
                feed.detach()

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
                    return self._send(200, live_page().encode("utf-8"), csp=LIVE_CSP)
                if path == "/live/stream":
                    return self._stream()
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
