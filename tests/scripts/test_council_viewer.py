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
    (h / "latest_strategy_tables.html").write_text("<p>not allowed</p>", encoding="utf-8")
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


def test_html_allowlist(root: Path):
    assert cv.html_file(root, "rules_council_summary.html") == b"<p>summary</p>"
    assert cv.html_file(root, "megu_cycles_50.html") == b"<p>megu</p>"
    assert cv.html_file(root, "latest_strategy_tables.html") is None
    assert cv.html_file(root, "../../../../.env") is None
    assert cv.html_file(root, "rules_council_missing.html") is None


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
    assert "latest_strategy_tables.html" not in out


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
        assert get("/html/operations/final/rules_council_summary.html") == 200
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
        c.request("GET", "/council/protocol.md")
        r = c.getresponse()
        assert "default-src 'none'" in (r.getheader("Content-Security-Policy") or "")
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
        c.request("GET", "/", headers={"Host": "evil.example:80"})
        assert c.getresponse().status == 403  # DNS rebinding
        for bad in ("/council/../../../../secret.md", "/council/%2e%2e/%2e%2e/%2e%2e/%2e%2e/secret.md",
                    "/council//etc/passwd", "/html/operations/final/../../../../.env",
                    "/html/operations/final/latest_strategy_tables.html", "/etc/passwd", "/.env"):
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
