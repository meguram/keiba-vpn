"""docs/html の日本語の階層（棚）の目次・まとめのページを生成する。

    python -m src.scripts.docs.build_docs_shelves          # 生成して書き出す（本の生成 build_docs_book からも先に呼ばれる）
    python -m src.scripts.docs.build_docs_shelves --check  # 書き出さずに、リンク切れと振り分け漏れだけを調べる

作るもの（どれも生成物。手で直さない）:
  docs/html/index.html                       全体の目次（棚ごとに各ページの 1 行の説明つき）
  docs/html/01_いまのベスト分析/目次.html     入口（5 テーマの結論の 1〜2 文）
  docs/html/01_いまのベスト分析/<テーマ>.html  まとめのページ（best_analysis.py。本の先頭の部にも載る）
  docs/html/02_本番の仕様/目次.html ほか、各棚の 目次.html

元の HTML は動かさない（振り分けは docs_shelves.py）。費用 0 円（手元のファイルと git log だけを読む）。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from src.scripts.docs import best_analysis as BA
from src.scripts.docs import docs_shelves as S

REPO = BA.REPO
DOCS = REPO / "docs"
HTML_DIR = DOCS / "html"
META_DIR = REPO / "data" / "meta"
GEN_MARK = '<meta name="generator" content="build_docs_shelves">'
TOC_NAME = "目次.html"
TOP = HTML_DIR / "index.html"
REQUIREMENTS = HTML_DIR / "02_本番の仕様" / "要件定義書.html"
BEST_PART_TITLE = "いまのベスト分析"
BEST_PART_LEAD = "データ分析家が最初に見る所。元の記録（json・md・HTML）から生成した、今の最良の結論・数字・根拠・限界。"

# 01_いまのベスト分析/ のページ（slug, ファイル名, 本の目次の説明）。本の PARTS もここから作る
BEST_PAGES: list[tuple[str, str, str]] = [
    ("best_graded", "重賞の好走データ予想.html", "重賞のデータ分析型の予想の今の結論と、評価の回ごとの結果の年表（生成）。"),
    ("best_course", "通常レースの好走データ抽出.html", "全レース版（コース × クラス）の好走データ抽出の今の結論と、版ごとの結果の年表（生成）。"),
    ("best_pedigree", "血統分析のまとめ.html", "血統の各調査の結論を 1 枚に並べたもの（生成）。"),
    ("best_model", "馬の予測モデル.html", "今の本番のモデルの版・採用時の精度・更新の年表・改善中の候補（生成）。"),
    ("best_bet", "推奨馬券の最適化.html", "推奨馬券の凍結した買い方と test 2025 の結果・サイクルの年表（生成）。"),
]


@dataclass
class Item:
    key: str            # 本の slug、または data/meta のパス
    title: str
    blurb: str
    kind: str
    status: str
    path: Path
    shelf: str
    part: str = ""
    updated: str = ""


def best_dir() -> Path:
    return HTML_DIR / S.SHELF["best"].dir


def shelf_dir(key: str) -> Path:
    return HTML_DIR / S.SHELF[key].dir


def esc(s) -> str:
    return BA.esc(s)


def rel(target: Path, out_dir: Path) -> str:
    return BA.rel(target, out_dir)


def is_generated(p: Path) -> bool:
    """このスクリプトが作るページか（本に載せない目次と、本に載せるまとめのページ）。"""
    try:
        head = p.read_text(encoding="utf-8", errors="ignore")[:2000]
    except OSError:
        return False
    return GEN_MARK in head


def nav_pages() -> set[Path]:
    """本に載せない目次のページ（index.html と各棚の 目次.html）。"""
    out = {TOP.resolve()}
    out |= {p.resolve() for p in HTML_DIR.rglob(TOC_NAME)}
    return out


# ── 更新日（git log を 1 回だけ流して、ファイルごとの最後のコミット日を取る） ─────────

_DATES: dict[str, str] | None = None


def _git_dates() -> dict[str, str]:
    global _DATES
    if _DATES is None:
        _DATES = {}
        try:
            out = subprocess.run(["git", "-c", "core.quotePath=false", "log", "--format=@%cd", "--date=format:%Y-%m-%d", "--name-only", "--", "docs", "data/meta"],
                                 capture_output=True, text=True, cwd=REPO, timeout=120).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        d = ""
        for line in out.splitlines():
            if line.startswith("@"):
                d = line[1:]
            elif line and line not in _DATES:
                _DATES[line] = d
    return _DATES


def updated(p: Path) -> str:
    try:
        key = str(p.resolve().relative_to(REPO.resolve()))
    except ValueError:
        key = str(p)
    return _git_dates().get(key) or BA.NO_RECORD   # git に無い（未コミットの）ファイルは「記録なし」（mtime は環境ごとに違うので使わない）


# ── 振り分け ──────────────────────────────────────────────────────────────

def chapter_shelf(ch, part_title: str) -> str:
    if ch.slug in S.OVERRIDE:
        return S.OVERRIDE[ch.slug]
    if ch.status in S.OLD_STATUS:
        return "old"
    if part_title == BEST_PART_TITLE:
        return "best"
    if part_title not in S.PART_DEFAULT:   # 本の生成は止めない（漏れは test_build_docs_shelves が検出する）
        _warn(f"本の部「{part_title}」の既定の棚が docs_shelves.PART_DEFAULT に無い（章 {ch.slug}）→ {S.SHELF[S.UNKNOWN_SHELF].title}")
        return S.UNKNOWN_SHELF
    return S.PART_DEFAULT[part_title]


def _warn(msg: str) -> None:
    print("[警告] " + msg, file=sys.stderr)


def meta_html_files() -> list[Path]:
    """git 管理の data/meta の研究の HTML（未コミットのものは CI で無いのでリンクしない）。"""
    try:
        out = subprocess.run(["git", "-c", "core.quotePath=false", "ls-files", "--", "data/meta"], capture_output=True, text=True, cwd=REPO, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    return sorted(REPO / f for f in out.splitlines() if f.endswith(".html") and (REPO / f).exists())


def meta_shelf(p: Path) -> str:
    sub = p.relative_to(META_DIR).parts[0] if len(p.relative_to(META_DIR).parts) > 1 else ""
    if sub not in S.META_DIR_SHELF:
        _warn(f"data/meta/{sub}/ の棚が docs_shelves.META_DIR_SHELF に無い（{p.name}）→ {S.SHELF[S.UNKNOWN_SHELF].title}")
        return S.UNKNOWN_SHELF
    return S.META_DIR_SHELF[sub]


def _meta_title_blurb(p: Path) -> tuple[str, str]:
    from src.scripts.docs.build_docs_book import derive_blurb, extract
    raw = p.read_text(encoding="utf-8", errors="ignore")[:300_000]
    t = re.search(r"<title>(.*?)</title>", raw, re.S)
    title = re.sub(r"\s+—.*$", "", BA.text_of(t.group(1))) if t else p.stem
    try:
        blurb = derive_blurb(extract(raw)[2])
    except Exception:  # noqa: BLE001 — 説明が作れなくても目次は作る
        blurb = ""
    return title, blurb


def collect_items() -> list[Item]:
    from src.scripts.docs import build_docs_book as B
    items: list[Item] = []
    for part in B.PARTS:
        for ch in part.chapters:
            items.append(Item(ch.slug, "", ch.blurb, ch.kind, ch.status, ch.path, chapter_shelf(ch, part.title), part.title))
    for p in meta_html_files():
        title, blurb = _meta_title_blurb(p)
        items.append(Item(str(p.relative_to(REPO)), title, blurb, "結果", "", p, meta_shelf(p), "data/meta"))
    for it in items:
        if not it.title:
            it.title = _title_of(it.path) or it.key
        if not it.blurb and it.path.suffix == ".html" and it.path.exists():
            from src.scripts.docs.build_docs_book import derive_blurb, extract
            it.blurb = derive_blurb(extract(it.path.read_text(encoding="utf-8", errors="ignore"))[2])
        it.updated = updated(it.path)
    return items


def _title_of(p: Path) -> str:
    if not p.exists():
        return ""
    raw = p.read_text(encoding="utf-8", errors="ignore")[:20000]
    if p.suffix == ".md":
        m = re.search(r"^#\s+(.+)$", raw, re.M)
        return m.group(1).strip() if m else p.stem
    m = re.search(r"<h1\b[^>]*>(.*?)</h1>", raw, re.S) or re.search(r"<title>(.*?)</title>", raw, re.S)
    return re.sub(r"\s+—.*$", "", BA.text_of(m.group(1))) if m else p.stem


# ── ページの組み立て ───────────────────────────────────────────────────────

CSS = """
:root{--bg:#fff;--fg:#1f2430;--mut:#5b6474;--line:#d9dee7;--acc:#2456c9;--chip:#eef2fa;--warn:#a15c00}
@media (prefers-color-scheme:dark){:root{--bg:#141821;--fg:#e6e9ef;--mut:#9aa3b2;--line:#2c3342;--acc:#7aa9ff;--chip:#1f2635;--warn:#f0b35a}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.7 system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px 16px 64px}
a{color:var(--acc)}h1{font-size:1.6em;margin:.4em 0}h2{margin-top:1.8em;border-bottom:1px solid var(--line);padding-bottom:.2em}
.crumb{font-size:.9em;color:var(--mut)}.meta,.src{color:var(--mut);font-size:.88em}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:.5em 0;font-size:.92em}
th,td{border:1px solid var(--line);padding:4px 8px;vertical-align:top;text-align:left}th{background:var(--chip)}
.chip{display:inline-block;background:var(--chip);border-radius:4px;padding:0 6px;font-size:.82em;margin-right:4px;white-space:nowrap}
.old{color:var(--warn)}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.card{border:1px solid var(--line);border-radius:8px;padding:10px 14px}.card h3{margin:.2em 0}
details{margin:.6em 0}summary{cursor:pointer}.mdsec{border-left:3px solid var(--line);padding-left:12px;margin:.6em 0}
"""


def page_html(title: str, out: Path, crumbs: list[tuple[str, Path | None]], body: str, sources: str = "") -> str:
    out_dir = out.parent
    cr = " ／ ".join(f'<a href="{esc(rel(p, out_dir))}">{esc(t)}</a>' if p else esc(t) for t, p in crumbs)
    return (f'<!doctype html>\n<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n{GEN_MARK}\n'
            f"<title>{esc(title)} — keiba-vpn docs</title><style>{CSS}</style></head><body><main>\n"
            f'<nav class="crumb">{cr}</nav>\n<h1>{esc(title)}</h1>\n'
            f'<p class="meta">生成物（<code>python -m src.scripts.docs.build_docs_shelves</code>。手で直さない）。{sources}</p>\n'
            f"{body}\n</main></body></html>\n")


def _badges(it: Item) -> str:
    out = f'<span class="chip">{esc(it.kind)}</span>'
    st = {"legacy": "旧設計", "snapshot": "時点の記録", "history": "決定の履歴"}.get(it.status)
    if st:
        out += f'<span class="chip old">{st}</span>'
    return out


def _item_row(it: Item, out_dir: Path, here: str) -> list[str]:
    where = "" if it.shelf == here else f'<br><span class="src">（置き場: {esc(S.SHELF[it.shelf].title)}）</span>'
    return [esc(it.updated), f'<a href="{esc(rel(it.path, out_dir))}">{esc(it.title)}</a>{where}', _badges(it), esc(it.blurb)]


def _extra_row(e: S.Extra, out_dir: Path) -> list[str] | None:
    p = REPO / e.path
    if not p.exists():
        _warn(f"docs_shelves.GROUPS の {e.path} が無い")
        return None
    return [esc(updated(p)), f'<a href="{esc(rel(p, out_dir))}">{esc(e.title)}</a>', '<span class="chip">md・設定</span>',
            esc(e.blurb or e.path)]


HEAD = ["更新日", "文書", "種類", "説明"]


def _sorted(items: list[Item]) -> list[Item]:
    return sorted(items, key=lambda i: i.updated or "", reverse=True)


def config_citations(items: list[Item]) -> list[tuple[Item, list[str]]]:
    """本番の設定（config/ の json）が根拠として名前を挙げる章。"""
    names = {it.path.name: it for it in items if it.path.suffix == ".html" and it.shelf != "best"}
    hits: dict[str, set[str]] = {}
    for f in sorted((REPO / "config").rglob("*.json")):
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name in set(re.findall(r"([A-Za-z0-9_]+\.html)", txt)):
            if name in names:
                hits.setdefault(name, set()).add(str(f.relative_to(REPO)))
    return [(names[n], sorted(v)) for n, v in sorted(hits.items(), key=lambda kv: names[kv[0]].title)]


def build_shelf_page(key: str, items: list[Item]) -> tuple[Path, str]:
    sh = S.SHELF[key]
    out = shelf_dir(key) / TOC_NAME
    out_dir = out.parent
    by_key = {it.key: it for it in items}
    mine = [it for it in items if it.shelf == key]
    used: set[str] = set()
    parts = [f"<p>{esc(sh.lead)}</p>"]
    groups = list(S.GROUPS.get(key, []))
    if key == "old":
        for st, label, lead in [("legacy", "旧設計", "一部が今の実装と違う設計。各ページの冒頭・本の注記に今の正がある。"),
                                ("snapshot", "時点の記録", "ある時点の状態の記録。数字は当時のもの。"),
                                ("history", "決定の履歴", "決定記録（書き換えず、古くなったものは注記を足す）。")]:
            groups.append(S.Group(label, lead, [it.key for it in mine if it.status == st]))
    for g in groups:
        rows = []
        for slug in g.slugs:
            it = by_key.get(slug)
            if it is None:   # 章が消えた・改名された。生成は止めない（test_build_docs_shelves が検出する）
                _warn(f"docs_shelves.GROUPS の「{g.title}」にある章 {slug} が本の PARTS に無い")
                continue
            rows.append(it)
            used.add(slug)
        body_rows = [_item_row(it, out_dir, key) for it in _sorted(rows)] + [r for r in (_extra_row(e, out_dir) for e in g.extras) if r]
        if not body_rows:
            continue
        cur = next((it for it in _sorted(rows) if it.status not in S.OLD_STATUS), None)
        lead = esc(g.lead) + (f'（いちばん新しく更新された現行の文書: <a href="{esc(rel(cur.path, out_dir))}">{esc(cur.title)}</a>、{esc(cur.updated)}）' if cur else "")
        parts.append(f"<h2>{esc(g.title)}</h2><p>{lead}</p>" + BA.table(HEAD, body_rows, raw=True))
    rest = [it for it in mine if it.key not in used]
    if rest:
        parts.append(f"<h2>{'その他' if groups else '文書の一覧'}</h2>" + BA.table(HEAD, [_item_row(it, out_dir, key) for it in _sorted(rest)], raw=True))
    if key == "spec":
        cites = config_citations(items)
        if cites:
            parts.append("<h2>本番の設定が根拠に挙げる調査</h2><p>config/ の設定ファイルが、根拠として名前を挙げている調査のページ（自動で抜き出し）。"
                         "本番の規則・係数がどの調査から来たかを辿れる。</p>"
                         + BA.table(["文書", "置き場", "挙げている設定"],
                                    [[f'<a href="{esc(rel(it.path, out_dir))}">{esc(it.title)}</a>', esc(S.SHELF[it.shelf].title),
                                      esc("・".join(fs))] for it, fs in cites], raw=True))
        parts.insert(1, f'<p>モデルの今の版と採用時の精度は <a href="{esc(rel(best_dir() / "馬の予測モデル.html", out_dir))}">いまのベスト分析 / 馬の予測モデル</a>。</p>')
    crumbs = [("目次", TOP)]
    if key.startswith("res-"):
        crumbs.append((S.RESEARCH_TITLE, HTML_DIR / S.RESEARCH_DIR / TOC_NAME))
    crumbs.append((sh.title, None))
    return out, page_html(sh.title, out, crumbs, "\n".join(parts))


def build_research_top(items: list[Item]) -> tuple[Path, str]:
    out = HTML_DIR / S.RESEARCH_DIR / TOC_NAME
    rows = []
    for sh in S.SHELVES:
        if not sh.key.startswith("res-"):
            continue
        mine = [it for it in items if it.shelf == sh.key]
        latest = max((it.updated for it in mine), default="")
        rows.append([f'<a href="{esc(rel(shelf_dir(sh.key) / TOC_NAME, out.parent))}">{esc(sh.title)}</a>', esc(len(mine)), esc(latest), esc(sh.lead)])
    body = f"<p>{esc(S.RESEARCH_LEAD)}</p>" + BA.table(["テーマ", "文書の数", "最終更新", "説明"], rows, raw=True)
    return out, page_html(S.RESEARCH_TITLE, out, [("目次", TOP), (S.RESEARCH_TITLE, None)], body)


def build_best_pages(items: list[Item]) -> list[tuple[Path, str, BA.Page]]:
    d = best_dir()
    meta = [it.path for it in items if it.part == "data/meta"]
    graded = [p for p in meta if meta_shelf(p) == "res-graded"]
    course = [p for p in meta if meta_shelf(p) == "res-course"]
    ped = [p for p in meta if meta_shelf(p) == "res-pedigree" and "sire_lines" in p.parts]
    pages = [BA.build_graded(d, graded), BA.build_course(d, course), BA.build_pedigree(d, ped), BA.build_model(d), BA.build_bet(d)]
    order = {slug: i for i, (slug, _, _) in enumerate(BEST_PAGES)}
    pages.sort(key=lambda p: order[p.slug])
    out = []
    for pg in pages:
        path = d / pg.file
        srcs = ("元の記録（最終コミット日）: " + "・".join(f'<a href="{esc(rel(s, d))}">{esc(s.name)}</a>（{esc(updated(s))}）'
                                               for s in pg.sources if s.exists())) if pg.sources else ""
        crumbs = [("目次", TOP), (S.SHELF["best"].title, d / TOC_NAME), (pg.title, None)]
        out.append((path, page_html(pg.title, path, crumbs, f"<p>{esc(pg.blurb)}</p>\n" + pg.body, srcs), pg))
    return out


def build_best_entry(pages: list[BA.Page]) -> tuple[Path, str]:
    d = best_dir()
    out = d / TOC_NAME
    cards = []
    for pg in pages:
        cards.append(f'<div class="card"><h3><a href="{esc(pg.file)}">{esc(pg.title)}</a></h3>'
                     f'<p>{esc(pg.headline or BA.NO_RECORD)}</p><p class="src">{esc(pg.blurb)}</p></div>')
    body = (f"<p>{esc(S.SHELF['best'].lead)}</p><p>各ページは元の記録から生成する（数字を手で写さない）。担当のセッションが結論を "
            "md の <code>&lt;!-- best:begin --&gt;</code> 〜 <code>&lt;!-- best:end --&gt;</code> に書くと、そのまま写す。"
            f"記録に無い値は「{BA.NO_RECORD}」、まとめ途中のものは「{BA.WIP}」と出す。</p>"
            f'<div class="cards">{"".join(cards)}</div>')
    return out, page_html(S.SHELF["best"].title, out, [("目次", TOP), (S.SHELF["best"].title, None)], body)


def build_top(items: list[Item], pages: list[BA.Page]) -> tuple[Path, str]:
    out = TOP
    d = HTML_DIR
    parts = ['<p>keiba-vpn の文書の目次。元の文書は今の置き場のまま（生成スクリプトが書き出す先を変えないため）、'
             'ここと各棚の目次から辿る。全部を 1 冊にした本は <a href="book.html">book.html</a>。'
             f'要件定義書は <a href="{esc(rel(REQUIREMENTS, d))}">02_本番の仕様/要件定義書.html</a>。</p>']
    bd = best_dir()
    parts.append(f'<h2><a href="{esc(rel(bd / TOC_NAME, d))}">01 いまのベスト分析</a></h2><p>{esc(S.SHELF["best"].lead)}</p>')
    parts.append(BA.table(["テーマ", "今の結論（元の記録から）"],
                          [[f'<a href="{esc(rel(bd / pg.file, d))}">{esc(pg.title)}</a>', esc(pg.headline or BA.NO_RECORD)] for pg in pages], raw=True))
    for sh in S.SHELVES:
        if sh.key == "best":
            continue
        mine = _sorted([it for it in items if it.shelf == sh.key])
        num = sh.dir.split("_", 1)[0]
        label = f"{num} {S.RESEARCH_TITLE} / {sh.title}" if sh.key.startswith("res-") else f"{num} {sh.title}"
        parts.append(f'<h2><a href="{esc(rel(shelf_dir(sh.key) / TOC_NAME, d))}">{esc(label)}</a>（{len(mine)}）</h2><p>{esc(sh.lead)}</p>')
        rows = [_item_row(it, d, sh.key) for it in mine]
        parts.append(f"<details{' open' if sh.key in ('spec',) else ''}><summary>{len(mine)} 件の一覧</summary>" + BA.table(HEAD, rows, raw=True) + "</details>")
    return out, page_html("keiba-vpn 文書の目次", out, [("目次", None)], "\n".join(parts))


def build_all() -> dict[Path, str]:
    """生成するページ（パス → HTML）。書き出しはしない。"""
    items = collect_items()
    best = build_best_pages(items)
    pages = [pg for _, _, pg in best]
    out: dict[Path, str] = {p: h for p, h, _ in best}
    for p, h in [build_best_entry(pages), build_top(items, pages), build_research_top(items)] + \
            [build_shelf_page(sh.key, items) for sh in S.SHELVES if sh.key != "best"]:
        out[p] = h
    return out


def broken_links(pages: dict[Path, str]) -> list[str]:
    bad = []
    for p, h in pages.items():
        for url in re.findall(r'\s(?:href|src)="([^"#?]+)', h):
            if re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*:", url) or url.startswith("/"):
                continue
            from urllib.parse import unquote
            target = os.path.normpath(os.path.join(p.parent, unquote(url)))
            if not os.path.exists(target) and Path(target) not in pages:
                bad.append(f"{p.relative_to(REPO)} → {url}")
    return bad


def write_all() -> dict[Path, str]:
    pages = build_all()
    for p, h in pages.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists() or p.read_text(encoding="utf-8") != h:
            p.write_text(h, encoding="utf-8")
    return pages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="書き出さずにリンク切れだけを調べる")
    args = ap.parse_args()
    pages = build_all() if args.check else write_all()
    bad = broken_links(pages)
    for b in bad:
        print("リンク切れ:", b)
    print(f"{len(pages)} ページ{'（書き出さず）' if args.check else ''}、リンク切れ {len(bad)} 件")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
