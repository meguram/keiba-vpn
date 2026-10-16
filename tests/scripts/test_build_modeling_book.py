"""モデリング総覧(build_modeling_book)の整合テスト。modeling/ に HTML を足したのに本に載せ忘れる、を検出する。"""

from __future__ import annotations

from src.scripts.docs import build_modeling_book as B


def test_every_modeling_html_belongs_to_exactly_one_chapter():
    files = {p.stem for p in B.CHAPTERS_DIR.glob("*.html")}
    assert {p.name for p in B.MODELING_DIR.glob("*.html")} == {"book.html"}, "modeling/ 直下には book.html だけを置く（章の HTML は chapters/）"
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
    assert out.count('class="bk-chapter') == len(B.CHAPTERS) + 1      # + 用語集（付録）
    assert 'id="ch-glossary"' in out and 'href="#ch-glossary"' in out


def test_navigation_does_not_use_smooth_scroll_or_scrollintoview():
    # 長いページで『滑らかなスクロール』や scrollIntoView(目次側) を使うと、ジャンプが途中で打ち消されて目的の章に着かなかった（2026-10-16）
    assert "scroll-behavior: smooth" not in B.SHELL_CSS and "scroll-behavior:smooth" not in B.SHELL_CSS
    assert "scrollIntoView" not in B.SHELL_JS
    assert "scrollRestoration" in B.SHELL_JS


def test_sidebar_has_three_levels_and_a_toggle_per_chapter():
    out, _ = B.build()
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
    out, _ = B.build()
    for t in GL.TERMS:
        assert f'id="gl-{t.id}"' in out, t.id
    import re
    for gid in set(re.findall(r'class="gl-term" data-gl="([a-z0-9]+)"', out)):
        assert f'id="gl-{gid}"' in out, f"押せる用語 {gid} の解説が用語集に無い"
    assert out.count('class="gl-term"') > 100, "本文にほとんど注釈がついていない"
    # 用語ボタンが <a>・<code>・見出しの中に入って壊れていない
    assert not re.search(r"<a\b[^>]*>[^<]*<button", out) and not re.search(r"<code>[^<]*<button", out)
