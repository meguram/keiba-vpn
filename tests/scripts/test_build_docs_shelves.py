"""docs/html の日本語の階層（棚）の目次・まとめのページ（build_docs_shelves）のテスト。

- docs/html の全 HTML と data/meta の研究の HTML が、ちょうど 1 つの棚に入る（載せ忘れ・二重を防ぐ）
- 生成するページのリンクが切れていない
- 01_いまのベスト分析 のページが元の記録から作られ、記録に無い値は「記録なし」と出る
- 要件定義書のスキルが、新しい置き場（02_本番の仕様/要件定義書.html）に書く
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scripts.docs import best_analysis as BA
from src.scripts.docs import build_docs_book as B
from src.scripts.docs import build_docs_shelves as G
from src.scripts.docs import docs_shelves as S

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def items():
    return G.collect_items()


@pytest.fixture(scope="module")
def pages():
    return G.build_all()


def test_every_html_belongs_to_exactly_one_shelf(items):
    keys = [it.key for it in items]
    assert len(keys) == len(set(keys)), "同じ文書が 2 回載っている"
    paths = [it.path.resolve() for it in items]
    assert len(paths) == len(set(paths)), "同じファイルが 2 回載っている"
    assert all(it.shelf in S.SHELF for it in items)
    on_disk = {p.resolve() for p in G.HTML_DIR.rglob("*.html") if p.name != B.BOOK_NAME} - G.nav_pages()
    shelved = {p for p in paths if G.HTML_DIR.resolve() in p.parents}
    assert on_disk == shelved, f"棚に無い: {sorted(map(str, on_disk - shelved))[:5]} / ファイルが無い: {sorted(map(str, shelved - on_disk))[:5]}"
    meta = {p.resolve() for p in G.meta_html_files()}
    assert meta and meta <= set(paths), "data/meta の研究の HTML が棚に無い"


def test_shelf_tables_point_to_real_chapters_and_files():
    slugs = set(B.CHAPTERS)
    assert not set(S.OVERRIDE) - slugs, f"OVERRIDE に本に無い章: {sorted(set(S.OVERRIDE) - slugs)}"
    assert set(S.OVERRIDE.values()) <= set(S.SHELF)
    for key, groups in S.GROUPS.items():
        assert key in S.SHELF
        for g in groups:
            assert not set(g.slugs) - slugs, f"{g.title}: 本に無い章 {sorted(set(g.slugs) - slugs)}"
            for e in g.extras:
                assert (ROOT / e.path).exists(), f"{g.title}: {e.path} が無い"
    for part in B.PARTS:
        assert part.title in S.PART_DEFAULT or part.title == G.BEST_PART_TITLE, f"部「{part.title}」の既定の棚が無い"
    subs = {p.relative_to(G.META_DIR).parts[0] for p in G.meta_html_files() if len(p.relative_to(G.META_DIR).parts) > 1}
    assert subs <= set(S.META_DIR_SHELF), f"data/meta のフォルダの棚が docs_shelves.META_DIR_SHELF に無い: {sorted(subs - set(S.META_DIR_SHELF))}"
    assert not [p for p in G.meta_html_files() if len(p.relative_to(G.META_DIR).parts) == 1], "data/meta 直下の HTML は棚が決まらない"


def test_generated_pages_have_no_broken_links_and_are_committed(pages):
    assert not G.broken_links(pages), G.broken_links(pages)[:10]
    for p, html in pages.items():
        assert G.GEN_MARK in html and "手で直さない" in html
        assert p.exists(), f"{p.relative_to(ROOT)} が無い（python -m src.scripts.docs.build_docs_shelves で作って commit する）"
    assert G.TOP in pages and all((G.shelf_dir(s.key) / G.TOC_NAME) in pages for s in S.SHELVES)
    for slug, file, _ in G.BEST_PAGES:
        assert (G.best_dir() / file) in pages
        assert B.CHAPTERS[slug].path.resolve() == (G.best_dir() / file).resolve(), "本の先頭の部とまとめのページが対応していない"


def test_nav_pages_are_not_in_the_book_but_best_pages_are():
    files = B.all_source_files()
    assert G.TOP.resolve() not in files
    assert not any(p.name == G.TOC_NAME for p in files)
    assert all((G.best_dir() / f).resolve() in files for _, f, _ in G.BEST_PAGES)
    assert B.PARTS[0].title == G.BEST_PART_TITLE


def test_requirements_doc_moved_and_skill_writes_the_new_place():
    assert G.REQUIREMENTS.exists()
    assert B.CHAPTERS["requirements"].path.resolve() == G.REQUIREMENTS.resolve()
    rel = "docs/html/02_本番の仕様/要件定義書.html"
    skill = ROOT / ".claude/skills/update-requirements"
    for f in ["scripts/collect_state.py", "scripts/validate_requirements.py"]:
        txt = (skill / f).read_text(encoding="utf-8")
        assert f'ROOT / "{rel}"' in txt and 'ROOT / "docs/html/index.html"' not in txt, f
    md = (skill / "SKILL.md").read_text(encoding="utf-8")
    assert f"`{rel}` に書き出す" in md and "出力先 `" + rel in md


def test_best_block_and_result_headings(tmp_path):
    md = tmp_path / "x.md"
    md.write_text("# 題\n\n<!-- best:begin -->\n結論は A。\n<!-- best:end -->\n\n## 1. 先行の結果\n\n## 2. 結果（2026-11-13）\n本文\n"
                  "```\n## 3. 結果（コードの中）\n```\n### 2.1 開発期間の結果\n", encoding="utf-8")
    assert BA.best_block(md) == "結論は A。"
    heads = [t for _, t, _ in BA.result_headings(md)]
    assert heads == ["2. 結果（2026-11-13）", "2.1 開発期間の結果"]
    sec = BA.md_section(md, BA.result_headings(md)[0][2])
    assert sec.startswith("## 2. 結果") and "開発期間" in sec
    assert BA.best_block(tmp_path / "none.md") is None


def test_model_page_says_no_record_instead_of_guessing(tmp_path, monkeypatch):
    trials = tmp_path / "trials"
    trials.mkdir()
    (trials / "model_cycles_log.json").write_text(json.dumps({"conclusion": "（20 サイクルが終わってから書く）", "cycles": [], "baselines": []}), encoding="utf-8")
    monkeypatch.setattr(BA, "TRIALS", trials)
    monkeypatch.setattr(BA, "HTML_DIR", tmp_path / "html")
    pg = BA.build_model(tmp_path)
    assert BA.WIP in pg.body          # 結論が無い → 作業中
    assert "AUC" in pg.body and BA.NO_RECORD in pg.body
    assert "食い違い・要確認" in pg.body   # 採用の台帳が読めない → 版の突き合わせができない


def test_version_comment_is_split_per_version(tmp_path):
    f = tmp_path / "v.py"
    f.write_text('REFERENCE_VERSION = "3"    # 3: 軸を足す（2026-11-05）。2：全角の区切り（2026-11-03）。1: 最初\n', encoding="utf-8")
    ver, rows = BA._version_comment(f, "REFERENCE_VERSION")
    assert ver == "3"
    assert rows == [("3", "2026-11-05", "軸を足す"), ("2", "2026-11-03", "全角の区切り"), ("1", BA.NO_RECORD, "最初")]


def test_running_page_marks_wip_and_picks_up_the_pairs_html(tmp_path, monkeypatch):
    monkeypatch.setattr(BA, "RUN_PAIRS_HTML", tmp_path / "running_style_pairs.html")
    pg = BA.build_running(tmp_path)
    assert BA.WIP in pg.body                     # 調査の HTML が無いうちは「作業中」
    (tmp_path / "running_style_pairs.html").write_text(
        '<html><body><h1>t</h1><div class="box"><b>結論</b><ol><li>一つ目の結論。</li><li>二つ目。</li><li>三つ目。</li><li>四つ目。</li></ol></div></body></html>',
        encoding="utf-8")
    pg = BA.build_running(tmp_path)
    assert "一つ目の結論" in pg.body and "三つ目" in pg.body and "四つ目" not in pg.body   # できたら結論の初めの 3 項目とリンク
    assert 'href="running_style_pairs.html"' in pg.body
