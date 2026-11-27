"""会議の記録のビューア（src/scripts/docs/council_viewer.py）: パスの正規化・Markdown の変換・目次。"""

from __future__ import annotations

import http.client
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("markdown")

from src.scripts.docs import council_viewer as cv


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    c = tmp_path / "repo" / cv.COUNCIL_REL
    (c / "rc").mkdir(parents=True)
    (c / "protocol.md").write_text("# 進め方\n\n本文\n", encoding="utf-8")
    (c / "decisions.md").write_text(
        "# 決定\n\n## 2026-11-24 第 1 回の会議\n\n### 決定\n\n```\n## コードの中は見出しでない\n```\n\n## D1（2026-11-24）穴馬\n",
        encoding="utf-8")
    (c / "scoreboard.md").write_text("# 成績表\n\n説明\n\n| 議題 | 結果 |\n|---|---|\n| RC-2 | 0 |\n", encoding="utf-8")
    (c / "rc" / "R1_02_digest.md").write_text("# RC-2 の R1 のまとめ（2026-11-24）\n", encoding="utf-8")
    (c / "rc" / "R0_02_result.md").write_text("# R0 資料 RC-2 の結果（2026-11-24）\n", encoding="utf-8")
    (c / "rc" / "R0_10_late.md").write_text("# R0 資料 RC-10（2026-11-26）\n", encoding="utf-8")
    (c / "rc" / "R0_01_order.md").write_text("# R0 資料 RC-0（2026-11-24）\n", encoding="utf-8")
    h = tmp_path / "repo" / cv.HTML_REL
    h.mkdir(parents=True)
    (h / "rules_council_summary.html").write_text("<p>summary</p>", encoding="utf-8")
    (h / "megu_cycles_50.html").write_text("<p>megu</p>", encoding="utf-8")
    (h / "pedigree_model.html").write_text("<title>血統 &amp; モデル</title><p>pedigree</p>", encoding="utf-8")
    (h / "bad name.html").write_text("<p>bad</p>", encoding="utf-8")
    (h / "sub").mkdir()
    (h / "sub" / "inner.html").write_text("<p>inner</p>", encoding="utf-8")
    (h / "escape.html").symlink_to(tmp_path / "outside.html")
    (tmp_path / "outside.html").write_text("<p>outside</p>", encoding="utf-8")
    (h.parent / "running_axis_curves.html").write_text("<title>走りの軸</title><p>axis</p>", encoding="utf-8")
    (h / "alias.html").symlink_to(h / "megu_cycles_50.html")  # 中を指すリンクは元の名前で出す
    (h / "xss.html").write_text("<title>&lt;script&gt;x <b>y</b></title>", encoding="utf-8")
    (h.parent / "sub2").mkdir()
    (h.parent / "sub2" / "deep.html").write_text("<p>deep</p>", encoding="utf-8")
    (h.parent / "out_link.html").symlink_to(tmp_path / "outside.html")
    (tmp_path / "secret.md").write_text("# 秘密\n", encoding="utf-8")
    (tmp_path / "repo" / ".env").write_text("X=1\n", encoding="utf-8")
    return tmp_path / "repo"


# ── パスの正規化 ──

def test_safe_resolve_inside(root: Path):
    base = root / cv.COUNCIL_REL
    assert cv.safe_resolve(base, "protocol.md") == (base / "protocol.md").resolve()
    assert cv.safe_resolve(base, "rc/R0_01_order.md") is not None


@pytest.mark.parametrize("rel", [
    "../../../../secret.md", "rc/../../../../../secret.md", "/etc/passwd", "../../../.env",
    "", "rc/", "rc", "a\x00b.md", "..\\..\\x.md", "rc//R0_01_order.md", "missing.md",
])
def test_safe_resolve_rejects(root: Path, rel: str):
    assert cv.safe_resolve(root / cv.COUNCIL_REL, rel) is None


def test_safe_resolve_rejects_symlink_out(root: Path, tmp_path: Path):
    base = root / cv.COUNCIL_REL
    (base / "link.md").symlink_to(tmp_path / "secret.md")
    (base / "rc" / "dirlink").symlink_to(tmp_path, target_is_directory=True)
    assert cv.safe_resolve(base, "link.md") is None
    assert cv.safe_resolve(base, "rc/dirlink/secret.md") is None


def test_html_files(root: Path):
    assert cv.html_file(root, "rules_council_summary.html") == b"<p>summary</p>"
    assert cv.html_file(root, "megu_cycles_50.html") == b"<p>megu</p>"
    assert cv.html_file(root, "pedigree_model.html").endswith(b"<p>pedigree</p>")
    assert cv.html_file(root, "running_axis_curves.html", cv.HTML_OPS_REL).endswith(b"<p>axis</p>")
    for bad in ("../../../../.env", "rules_council_missing.html", "sub/inner.html", "bad name.html",
                "escape.html", "/etc/passwd", "x.htm", ".html", "megu_cycles_50.html\n"):
        assert cv.html_file(root, bad) is None, bad
    assert cv.html_file(root, "final/pedigree_model.html", cv.HTML_OPS_REL) is None
    assert cv.html_file(root, "sub2/deep.html", cv.HTML_OPS_REL) is None
    assert cv.html_file(root, "out_link.html", cv.HTML_OPS_REL) is None


# ── Markdown の変換 ──

def test_md_tables_and_code():
    out = cv.md_to_html("| a | b |\n|---|---|\n| 1 | 2 |\n\n```\nx < y\n```\n\n## 見出し\n")
    assert '<div class="scroll"><table>' in out and "<td>1</td>" in out
    assert "<pre><code>x &lt; y" in out
    assert 'id="見出し"' in out


def test_md_page_has_css_and_nav(root: Path):
    body = cv.md_page(root, "protocol.md")
    assert body and "--bg:#f7f8fb" in body and 'href="/"' in body and "<h1" in body
    assert "http://" not in body and "https://" not in body  # 外部の CDN を読まない
    assert cv.md_page(root, "../../../../secret.md") is None


# ── 目次 ──

def test_list_rc_order(root: Path):
    rows = cv.list_rc(root / cv.COUNCIL_REL)
    assert [r["name"] for r in rows] == ["R0_01_order.md", "R0_02_result.md", "R1_02_digest.md", "R0_10_late.md"]
    assert [r["agenda"] for r in rows] == [0, 2, 2, 10]  # R0_01 は見出しの RC-0
    assert rows[0]["date"] == "2026-11-24"


def test_decision_headings_match_toc_ids(root: Path):
    text = (root / cv.COUNCIL_REL / "decisions.md").read_text(encoding="utf-8")
    heads = cv.decision_headings(text)
    assert [t for t, _ in heads] == ["2026-11-24 第 1 回の会議", "D1（2026-11-24）穴馬"]
    rendered = cv.md_to_html(text)
    for _, slug in heads:
        assert f'id="{slug}"' in rendered


def test_index_lists_everything(root: Path):
    out = cv.index_html(root)
    assert "/council/protocol.md" in out and "/council/decisions.md#" in out
    assert "<td>RC-2</td>" in out  # 成績表の表
    assert "/council/rc/R0_10_late.md" in out
    assert "/html/operations/final/rules_council_summary.html" in out
    assert "/html/operations/final/megu_cycles_50.html" in out


def test_index_lists_output_html(root: Path):
    out = cv.index_html(root)
    sec = out[out.index("<h2>成果の HTML</h2>"):]
    assert out.index("<h2>関係の HTML</h2>") < out.index("<h2>成果の HTML</h2>")
    assert "/html/operations/final/pedigree_model.html" in sec and "血統 &amp; モデル" in sec
    assert 'href="/html/operations/running_axis_curves.html"' in sec and "走りの軸" in sec
    assert "pedigree_model.html" not in out[:out.index("<h2>成果の HTML</h2>")]  # 関係の HTML には出さない
    for bad in ("inner.html", "bad name.html", "bad%20name.html", "escape.html", "deep.html", "out_link.html"):
        assert bad not in out, bad
    assert 'href="/html/operations/final/alias.html"' in sec
    assert "&lt;script&gt;x y" in sec and "<script>" not in out
    head = out[:out.index("<h2>成果の HTML</h2>")]
    assert "running_axis_curves.html" not in head and "alias.html" not in head  # 関係の HTML は会議の分だけ


def test_index_shows_easy_summary_body_on_top(root: Path):
    (root / cv.COUNCIL_REL / "easy_summary.md").write_text("# やさしい説明\n\n試したことは **3 つ**。\n", encoding="utf-8")
    out = cv.index_html(root)
    assert "<strong>3 つ</strong>" in out
    assert out.index("やさしい説明") < out.index("<h2>議題（rc）</h2>")


def test_relative_links_point_into_council(root: Path):
    c = root / cv.COUNCIL_REL
    (c / "easy").mkdir()
    (c / "easy" / "RC-02.md").write_text("# RC-2 のやさしい説明\n\n[もどる](../easy_summary.md#表)\n", encoding="utf-8")
    (c / "easy_summary.md").write_text(
        "# 説明\n\n| 議題 | 説明 |\n|---|---|\n| RC-2 | [くわしく](easy/RC-02.md) |\n\n"
        "[外](../../secret.md) [web](https://example.com/) [頭](#x) [根](/council/protocol.md) "
        "[html](../../html/operations/final/rules_council_summary.html)\n", encoding="utf-8")
    out = cv.index_html(root)
    assert 'href="/council/easy/RC-02.md"' in out
    assert 'href="../../secret.md"' in out and 'href="https://example.com/"' in out and 'href="#x"' in out
    assert 'href="/council/protocol.md"' in out
    assert 'href="../../html/operations/final/rules_council_summary.html"' in out
    sub = cv.md_page(root, "easy/RC-02.md")
    assert sub and 'href="/council/easy_summary.md#' in sub


def test_index_picks_up_new_files(root: Path):
    (root / cv.COUNCIL_REL / "rc" / "R2_03_new.md").write_text("# 新しい資料（2026-11-30）\n", encoding="utf-8")
    assert "R2_03_new.md" in cv.index_html(root)


# ── サーバ（生の要求で .. を送る） ──

def test_server_rejects_traversal(root: Path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), cv.make_handler(root))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        def get(path: str) -> int:
            c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
            c.request("GET", path)
            return c.getresponse().status

        assert get("/") == 200
        assert get("/healthz") == 200
        assert get("/council/rc/R0_01_order.md") == 200
        assert get("/doc/rc/R0_01_order.md") == 200  # /doc/ は /council/ と同じ
        assert get("/doc/../../../../secret.md") == 404
        assert get("/html/operations/final/rules_council_summary.html") == 200
        assert get("/html/operations/final/pedigree_model.html") == 200
        assert get("/html/operations/running_axis_curves.html") == 200
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
        c.request("HEAD", "/html/operations/final/pedigree_model.html")
        r = c.getresponse()
        assert r.status == 200 and r.read() == b""
        assert (r.getheader("Content-Security-Policy") or "").startswith("sandbox")
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
        c.request("GET", "/council/protocol.md")
        r = c.getresponse()
        assert "default-src 'none'" in (r.getheader("Content-Security-Policy") or "")
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
        c.request("GET", "/", headers={"Host": "evil.example:80"})
        assert c.getresponse().status == 403  # DNS rebinding
        for bad in ("/council/../../../../secret.md", "/council/%2e%2e/%2e%2e/%2e%2e/%2e%2e/secret.md",
                    "/council//etc/passwd", "/html/operations/final/../../../../.env",
                    "/html/operations/final/sub/inner.html", "/html/operations/final/escape.html",
                    "/html/operations/final/bad%20name.html", "/html/operations/../../.env",
                    "/html/operations/%2e%2e/%2e%2e/.env", "/html/operations/final/", "/html/operations/",
                    "/html/operations/final", "/html/operations/sub2/deep.html", "/html/operations/out_link.html",
                    "/html/operations/final/megu_cycles_50.html%0A",
                    "/etc/passwd", "/.env"):
            assert get(bad) == 404, bad
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.mark.parametrize("host,ok", [
    ("127.0.0.1:23950", True), ("localhost:8090", True), ("localhost", True), ("[::1]:23950", True),
    ("evil.example", False), ("127.0.0.1.evil.example:23950", False), ("", False),
])
def test_host_allowed(monkeypatch, host: str, ok: bool):
    monkeypatch.delenv("KEIBA_COUNCIL_ALLOW_ANY_HOST", raising=False)
    assert cv.host_allowed(host) is ok


def test_default_root_is_repo():
    assert (cv.DEFAULT_ROOT / "config" / "ports.env").is_file()
    assert (cv.DEFAULT_ROOT / cv.COUNCIL_REL).is_dir()


# ── ライブの吹き出しの Markdown（会話の本文を安全に描く） ──

def test_safe_md_renders_markdown_like_docs():
    text = ("# 見出し\n**太字** と `a<b>&c`\n- 箇条 1\n  - 入れ子\n1. 番号\n2. 番号\n"
            "| 項目 | 値 |\n|---|---|\n| 回収率 | 98% |\n"
            "```python\nx = '<b>' & 1\n```\n> 引用\n\n[安全](https://example.com) [相対](docs/a.md)")
    out = cv.safe_md_to_html(text)
    assert "<h1>見出し</h1>" in out and "<strong>太字</strong>" in out
    assert "<code>a&lt;b&gt;&amp;c</code>" in out
    assert "<ul>" in out and "<li>入れ子</li>" in out and "<ol>" in out  # 段落の直後の箇条・2 字下げの入れ子も描く
    assert '<div class="scroll"><table>' in out and "<td>98%</td>" in out
    assert "x = &#x27;&lt;b&gt;&#x27; &amp; 1" in out or "x = '&lt;b&gt;' &amp; 1" in out
    assert "<blockquote>" in out
    assert 'href="https://example.com"' in out and 'href="docs/a.md"' in out and 'rel="noopener noreferrer"' in out


@pytest.mark.parametrize("bad", [
    "<script>alert(1)</script>",
    '<img src=x onerror="alert(1)">',
    '<div onclick="alert(1)">x</div>',
    "<iframe src=//evil></iframe>",
    "<svg/onload=alert(1)>",
])
def test_safe_md_escapes_raw_html(bad: str):
    out = cv.safe_md_to_html(f"前\n\n{bad}\n\n後 {bad}")
    assert "<script" not in out and "<img" not in out and "<div onclick" not in out and "<iframe" not in out and "<svg" not in out
    assert "&lt;" in out


@pytest.mark.parametrize("href", ["javascript:alert(1)", "JaVaScRiPt:alert(1)", "java&#115;cript:alert(1)",
                                  "data:text/html,<script>1</script>", "vbscript:x", " javascript:alert(1)"])
def test_safe_md_drops_dangerous_links(href: str):
    out = cv.safe_md_to_html(f"[押す]({href})")
    assert "href" not in out and "押す" in out


def test_safe_md_image_becomes_text():
    out = cv.safe_md_to_html('![図 <x>](http://evil/a.png "t")')
    assert "<img" not in out and "［画像: 図 &lt;x&gt;］" in out


def test_event_json_renders_text_once_and_keeps_tools_plain():
    e = {"seq": 1, "ts": "t", "speaker": "1 号（リーダー）", "cls": "s1", "kind": "text", "text": "<b>x</b> **y**",
         "stage": "R3", "topic": "R0-17", "decision": True}
    j = cv.event_json(e)
    assert j["html"] == "<p>&lt;b&gt;x&lt;/b&gt; <strong>y</strong></p>" and "text" not in j and j["decision"]
    assert e["html"] == j["html"]  # 2 回目からは描いた結果を使う
    t = cv.event_json({"seq": 2, "ts": "t", "speaker": "s", "cls": "sc", "kind": "tool", "text": "Bash: <x>",
                       "stage": "", "topic": "", "decision": False})
    assert t["text"] == "Bash: <x>" and "html" not in t


def test_live_page_is_inline_only():
    body = cv.live_page("abc")
    assert '<script nonce="abc">' in body and "src=" not in body.split("<script", 1)[1]
    assert "http://" not in body and "https://" not in body
    assert "innerHTML = e.html" in body  # 本文はサーバで描いた安全な HTML だけを入れる
    for n in cv.LIVE_LIMITS:
        assert f'<option value="{n}">' in body


@pytest.mark.parametrize("href", ["<javascript:alert(1)>", "java&#x09;script:alert(1)", "javascript&colon;alert(1)"])
def test_safe_md_drops_wrapped_dangerous_links(href: str):
    out = cv.safe_md_to_html(f"[押す]({href})\n\n[r]\n\n[r]: javascript:alert(2)")
    assert "javascript" not in out.lower().replace("javascript&amp;colon;", "").split("href")[-1] or "href" not in out
    assert 'href="javascript' not in out.lower() and "href=\"&lt;javascript" not in out.lower()


def test_modeling_final_html_served_and_listed(root: Path, tmp_path: Path):
    mdir = root / cv.HTML_MODEL_REL
    (mdir / "sub").mkdir(parents=True)
    (mdir / "ensemble_architecture.html").write_text("<title>構成図</title><p>ens</p>", encoding="utf-8")
    (mdir / "sub" / "deep.html").write_text("x", encoding="utf-8")
    (mdir / "out_link.html").symlink_to(tmp_path / "outside.html")
    (tmp_path / "outside.html").write_text("secret", encoding="utf-8")
    assert cv.html_file(root, "ensemble_architecture.html", cv.HTML_MODEL_REL).endswith(b"<p>ens</p>")
    for bad in ("sub/deep.html", "out_link.html", "../../../../.env", "bad name.html"):
        assert cv.html_file(root, bad, cv.HTML_MODEL_REL) is None, bad
    out = cv.index_html(root)
    sec = out[out.index("<h2>成果の HTML</h2>"):]
    assert "<code>docs/html/modeling/final/</code>" in sec
    assert 'href="/html/modeling/final/ensemble_architecture.html"' in sec and "構成図" in sec
    assert "deep.html" not in sec and "out_link.html" not in sec.split("modeling/final/")[1]
    srv = ThreadingHTTPServer(("127.0.0.1", 0), cv.make_handler(root))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        def get(path: str):
            c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
            c.request("GET", path)
            r = c.getresponse()
            r.read()
            return r

        r = get("/html/modeling/final/ensemble_architecture.html")
        assert r.status == 200 and (r.getheader("Content-Security-Policy") or "").startswith("sandbox")
        for bad in ("/html/modeling/final/sub/deep.html", "/html/modeling/final/out_link.html",
                    "/html/modeling/final/../../../.env", "/html/modeling/final/%2e%2e/%2e%2e/.env",
                    "/html/modeling/final/", "/html/modeling/final"):
            assert get(bad).status == 404, bad
    finally:
        srv.shutdown()
        srv.server_close()
