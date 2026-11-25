"""匿名に出すコラム（frontend/lib/learn.ts の PUBLIC_COLUMN_SLUGS）が、出してよい範囲に収まるか。

出してよいのは、成績（着順）と確定オッズを集計した率だけの記事（統括役の承認 2026-11-25）。
netkeiba 独自の情報（追い切り・厩舎コメント・指数など）・確定前のオッズ・区間ラップ・馬体重と、
「買い」・回収率の見せ方を含む記事は外す。記事の JSON とガイドの全文を語で確かめる。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLS = ROOT / "frontend" / "content" / "columns"
LEARN = ROOT / "frontend" / "lib" / "learn.ts"

BANNED = re.compile(
    r"調教(?!師)|追い切り|坂路|ウッド|乗り込み|コメント|談話|短評|指数|持ち時計|前日|T-\d+|発走前|"
    r"馬体重|ラップ|上がり|通過順|4角|回収率|期待値|買い|狙い|推奨"
)


def _slugs() -> list[str]:
    src = LEARN.read_text(encoding="utf-8")
    block = re.search(r"PUBLIC_COLUMN_SLUGS = \[(.*?)\] as const", src, re.S)
    assert block, "PUBLIC_COLUMN_SLUGS が見つからない"
    return re.findall(r'"([a-z0-9-]+)"', block.group(1))


def test_public_columns_exist_and_unique():
    slugs = _slugs()
    assert slugs and len(slugs) == len(set(slugs))
    for s in slugs:
        assert (COLS / f"{s}.json").is_file(), s


def test_public_columns_have_no_banned_terms():
    bad = {}
    for s in _slugs():
        text = (COLS / f"{s}.json").read_text(encoding="utf-8")
        g = COLS / "guides" / f"{s}.json"
        if g.is_file():
            text += g.read_text(encoding="utf-8")
        hits = sorted(set(BANNED.findall(text)))
        if hits:
            bad[s] = hits
    assert not bad, f"匿名に出すコラムに、出さない語がある: {bad}"


def test_public_columns_are_not_drafts():
    for s in _slugs():
        d = json.loads((COLS / f"{s}.json").read_text(encoding="utf-8"))
        assert not d.get("draft"), s


PROMO = re.compile(r"必勝|稼げる|稼ぐ|儲か|今すぐ買|推奨|勝負気配|絶対")


def test_learn_pages_and_terms_have_no_promotional_words():
    """用語集・使い方・入口の文言に、買うことを促す言い方が無い（docs/operations/wording_glossary.md）"""
    files = [LEARN, COLS / "guides" / "glossary.json",
             ROOT / "frontend" / "components" / "home" / "LearnEntry.tsx",
             *sorted((ROOT / "frontend" / "app" / "learn").rglob("page.tsx"))]
    found = {str(f.relative_to(ROOT)): sorted(set(PROMO.findall(f.read_text(encoding="utf-8"))) ) for f in files}
    bad = {k: v for k, v in found.items() if v}
    assert not bad, f"買うことを促す言い方がある: {bad}"
