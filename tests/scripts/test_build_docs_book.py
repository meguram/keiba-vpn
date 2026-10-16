"""ドキュメント総覧(build_docs_book)の整合テスト。docs/ に HTML を足したのに本に載せ忘れる、を検出する。"""

from __future__ import annotations

import re
from urllib.parse import unquote

from functools import lru_cache

from src.scripts.docs import build_docs_book as B


@lru_cache(maxsize=1)
def _build():
    return B.build()


def test_every_html_under_docs_belongs_to_exactly_one_chapter():
    chapters = [c for p in B.PARTS for c in p.chapters]
    slugs = [c.slug for c in chapters]
    assert len(slugs) == len(set(slugs)), "同じ slug の章がある"
    paths = [c.path.resolve() for c in chapters]
    assert len(paths) == len(set(paths)), "同じファイルが 2 回載っている"
    assert not list(B.MODELING_DIR.glob("*.html")), "modeling/ 直下には HTML を置かない（章は chapters/、本は docs/html/book.html）"
    files = B.all_source_files()
    assert files == set(paths), f"本に載っていない: {sorted(str(p) for p in files - set(paths))} / ファイルが無い: {sorted(str(p) for p in set(paths) - files)}"


def test_related_chapters_exist():
    for p in B.PARTS:
        for c in p.chapters:
            for s, _ in c.see:
                assert s in B.CHAPTERS, f"{c.slug} の関連章 {s} が無い"
            assert c.kind in B.KINDS
            for s, _ in c.now:
                assert s in B.CHAPTERS, f"{c.slug} の現行章 {s} が無い"
            assert c.status in ("", *B.STATUS), c.status
            assert not c.status or c.status == "history" or c.note, f"{c.slug}: 状態があるのに説明(note)が無い"


def test_curly_quotes_in_tags_are_repaired():
    assert B.fix_curly_quotes('<h2 id=”_1”>x “引用” y</h2>') == '<h2 id="_1">x “引用” y</h2>'


def test_ids_and_links_are_namespaced_per_chapter():
    body = '<h2 id="a">A</h2><a href="#a">self</a><a href="other.html#b">x</a><a href="other.html">y</a><a href="../ops/z.html">z</a><marker id="ah"/><line marker-end="url(#ah)"/>'
    out = B.namespace(body, "me", {"me", "other"})
    assert 'id="me--a"' in out and 'href="#me--a"' in out
    assert 'href="#other--b"' in out and 'href="#ch-other"' in out
    assert 'href="../ops/z.html"' in out          # 本の外のリンクは変えない
    assert 'id="me--ah"' in out and "url(#me--ah)" in out


def test_css_is_scoped_to_the_chapter():
    css = ":root{--a:1} body{color:red} h2, .x{margin:0} @media (max-width:600px){ table{width:100%} } @keyframes k{from{opacity:0}}"
    out = B.scope_css(css, ".css-t")
    assert ".css-t{--a:1}" in out and ".css-t{color:red}" in out
    assert ".css-t h2, .css-t .x{margin:0}" in out and ".css-t table{width:100%}" in out
    assert "@keyframes k" in out


def test_extract_drops_sidebar_and_nav_and_keeps_title():
    raw = ('<html><head><title>T — x</title><style>b{}</style></head><body><aside class="sidebar">SIDE</aside><main>'
           '<header class="hero"><h1>見出し</h1></header><article class=”html-body”><h1 id=”_1”>見出し</h1><h2 id=”_2”>節</h2><p>本文</p></article>'
           '<nav class="doc-nav"><a>前</a></nav><footer>F</footer></main></body></html>')
    title, css, body = B.extract(raw)
    assert title == "見出し" and "SIDE" not in body and "doc-nav" not in body and "<footer" not in body
    assert "<h2" in body and "本文" in body and "b{}" in css


def test_build_contains_every_chapter_and_toc():
    out, built = _build()
    assert set(built) == set(B.CHAPTERS)
    for slug in B.CHAPTERS:
        assert f'id="ch-{slug}"' in out and f'href="#ch-{slug}"' in out
    assert out.count('class="bk-chapter') == len(B.CHAPTERS) + 1      # + 用語集（付録）
    assert 'id="ch-glossary"' in out and 'href="#ch-glossary"' in out


def test_navigation_does_not_use_smooth_scroll_or_scrollintoview():
    # 長いページで『滑らかなスクロール』や scrollIntoView(目次側) を使うと、ジャンプが途中で打ち消されて目的の章に着かなかった（2026-10-16）
    assert "scroll-behavior: smooth" not in B.SHELL_CSS and "scroll-behavior:smooth" not in B.SHELL_CSS
    assert "scrollIntoView" not in B.SHELL_JS
    assert "scrollRestoration" in B.SHELL_JS


def test_sidebar_has_three_levels_and_a_toggle_per_chapter():
    out, _ = _build()
    assert out.count('class="part"') == len(B.ALL_PARTS) and out.count('class="tg"') == len(B.CHAPTERS) + 1
    assert out.count('class="secs"') == len(B.CHAPTERS) + 1 and 'id="bk-open"' in out and 'id="bk-close"' in out


# ── 用語集 ────────────────────────────────────────────────────────────────────

from src.scripts.docs import modeling_glossary as GL  # noqa: E402


def test_glossary_terms_are_well_formed():
    ids = [t.id for t in GL.TERMS]
    assert len(ids) == len(set(ids)), "用語の id が重複している"
    cats = {c for c, _ in GL.CATS}
    import re
    for t in GL.TERMS:
        assert re.fullmatch(r"[a-z0-9]+", t.id), t.id
        assert t.cat in cats and t.pats and t.short and t.name, t.id
        assert not t.diagram or t.diagram in GL.DIAGRAMS, f"{t.id}: 図 {t.diagram} が無い"
        for pat in t.pats:
            re.compile(pat)
    for k, f in GL.DIAGRAMS.items():
        svg = f()
        assert svg.startswith("<svg") and svg.endswith("</svg>") and " id=" not in svg, k    # 同じ図を何度も複製するので id を持たない


def test_glossary_see_links_point_to_real_chapters():
    for t in GL.TERMS:
        for s in t.see:
            assert s in B.CHAPTERS, f"用語 {t.id} の関連章 {s} が無い"


def test_annotate_wraps_first_use_only_and_skips_code_links_headings():
    used: dict = {}
    body = ('<p>OOF と OOF。<code>LogLoss</code> <a href="#x">AUC</a></p><h2>ECE</h2><svg><text>Brier</text></svg>'
            '<div class="mermaid">Harville</div><p>valid と validate、LogLoss、T-60、ewm2</p>')
    out = GL.annotate(body, used, "c1")
    assert out.count('data-gl="oof"') == 1 and out.count("OOF") == 2
    for skipped in ("ece", "brier", "harville", "auc"):
        assert f'data-gl="{skipped}"' not in out, skipped
    assert out.count('data-gl="logloss"') == 1 and "<code>LogLoss</code>" in out
    assert 'data-gl="valid"' in out and ">validate" not in out.replace("と validate", "")     # 単語の途中は押せるようにしない
    assert 'data-gl="tier"' in out and 'data-gl="ewm"' not in out
    assert used["oof"] == {"c1"}


def test_build_includes_glossary_with_every_term_and_buttons_resolve():
    out, _ = _build()
    for t in GL.TERMS:
        assert f'id="gl-{t.id}"' in out, t.id
    import re
    for gid in set(re.findall(r'class="gl-term" data-gl="([a-z0-9]+)"', out)):
        assert f'id="gl-{gid}"' in out, f"押せる用語 {gid} の解説が用語集に無い"
    assert out.count('class="gl-term"') > 100, "本文にほとんど注釈がついていない"
    # 用語ボタンが <a>・<code>・見出しの中に入って壊れていない
    assert not re.search(r"<a\b[^>]*>[^<]*<button", out) and not re.search(r"<code>[^<]*<button", out)


# ── 全体を 1 冊にしたことで増えた取り決め ───────────────────────────────────────

def test_namespace_resolves_links_across_directories(tmp_path):
    a = str(tmp_path / "design" / "sub" / "A.html")
    b = str(tmp_path / "data" / "B.html")
    body = '<a href="../../data/B.html#x">b</a><a href="../../data/B.html">b2</a><a href="../../src/m.py">code</a><a href="https://e.com/x.html">ext</a><h2 id="s">s</h2>'
    out = B.namespace(body, "A", {"A", "B"}, a, {b: "B"}, str(tmp_path / "html"))
    assert 'href="#B--x"' in out and 'href="#ch-B"' in out
    assert 'href="https://e.com/x.html"' in out
    assert 'href="../src/m.py"' in out and 'href="../../src/m.py"' not in out      # 本の位置(html/)から見た相対パスに直す
    assert 'id="A--s"' in out


def test_duplicate_ids_in_a_source_page_are_made_unique():
    out = B.namespace('<marker id="ah"/><marker id="ah"/>', "c", {"c"})
    assert 'id="c--ah"' in out and 'id="c--ah__2"' in out


def test_book_has_no_duplicate_ids_dead_anchors_or_broken_links():
    import collections
    import os
    out, _ = _build()
    out = re.sub(r"<script\b.*?</script>", "", out, flags=re.S)       # script 内の文字列は対象外
    ids = re.findall(r'\sid="([^"]+)"', out)
    assert not [k for k, v in collections.Counter(ids).items() if v > 1], "id が重複している"
    idset = set(ids)
    dead = [h for h in re.findall(r'\shref="#([^"]*)"', out) if h and unquote(h) not in idset]
    assert not dead, f"本の中のリンク先が無い: {dead[:10]}"
    broken = set()
    for h in re.findall(r'\s(?:href|src)="([^"#?]+)', out):
        if re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*:", h) or h.startswith("/"):
            continue
        if not os.path.exists(os.path.normpath(os.path.join(B.HTML_DIR, unquote(h)))):
            broken.add(h)
    assert not broken, f"本の外へのリンクが切れている: {sorted(broken)[:10]}"


def test_book_does_not_depend_on_external_cdn():
    out, _ = _build()
    assert "cdn.jsdelivr" not in out and "fonts.googleapis" not in out
    assert (B.HTML_DIR / "assets" / "mermaid.min.js").is_file() and 'src="assets/mermaid.min.js"' in out


def test_legacy_chapters_say_what_is_different_and_where_the_current_one_is():
    for p in B.PARTS:
        for c in p.chapters:
            if c.status == "legacy":
                assert c.note and c.now, f"{c.slug}: 旧設計には『異なる点』と『現行の章』が要る"
