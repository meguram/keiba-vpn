"""モデリング総覧(build_modeling_book)の整合テスト。modeling/ に HTML を足したのに本に載せ忘れる、を検出する。"""

from __future__ import annotations

from src.scripts.docs import build_modeling_book as B


def test_every_modeling_html_belongs_to_exactly_one_chapter():
    files = {p.stem for p in B.MODELING_DIR.glob("*.html")} - {"book"}
    slugs = [c.slug for p in B.PARTS for c in p.chapters]
    assert len(slugs) == len(set(slugs)), "同じ章が 2 回載っている"
    assert files == set(slugs), f"本に載っていない: {files - set(slugs)} / ファイルが無い: {set(slugs) - files}"


def test_related_chapters_exist():
    for p in B.PARTS:
        for c in p.chapters:
            for s, _ in c.see:
                assert s in B.CHAPTERS, f"{c.slug} の関連章 {s} が無い"
            assert c.kind in B.KINDS


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
    out, built = B.build()
    assert set(built) == set(B.CHAPTERS)
    for slug in B.CHAPTERS:
        assert f'id="ch-{slug}"' in out and f'href="#ch-{slug}"' in out
    assert out.count('class="bk-chapter') == len(B.CHAPTERS)
