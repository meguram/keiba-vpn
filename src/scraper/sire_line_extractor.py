"""
netkeiba の blood_table HTML から種牡馬の系統名（○○系）を抽出するユーティリティ。

netkeiba の horse/ped ページでは blood_table の <td> セル内に
    <a href="/horse/sire/...">種牡馬名</a>
    <br>サンデーサイレンス系          ← ← ← この行
の形式で系統名が含まれる場合がある。
セル内にない場合は空文字を返す。
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

# 「○○系」として認識する正規表現
# 2～28 文字（数字・改行を含まない文字）で終わりが "系"
_SIRE_LINE_RE = re.compile(r"^[^\d\n\r]{2,28}系$")

# rowspan → 世代 マッピング（parse_blood_table_5gen と同一）
_GEN_ROWSPAN = {32: 1, 16: 2, 8: 3, 4: 4, 2: 5}


def _td_sire_line(td) -> str:
    """<td> セルから系統名テキストを抽出する。

    リンク要素のテキストを除いた残りのテキストを走査し、
    「○○系」パターンにマッチする最初の文字列を返す。
    """
    link = td.select_one("a")
    link_text = link.get_text(strip=True) if link else ""

    # セル内テキスト全体からリンク文字列を除去して候補を集める
    full_text = td.get_text(separator="\n")
    for raw in full_text.splitlines():
        candidate = raw.strip()
        if not candidate or candidate == link_text:
            continue
        if _SIRE_LINE_RE.match(candidate):
            return candidate
    return ""


def _td_sex(td) -> str:
    """<td> セル内の全リンク href から性別を判定する。

    /horse/sire/ または sire_horse → 牡
    /broodmare / /mare/ / broodmare_horse → 牝
    どちらも見つからない場合は "" (不明)
    """
    for a in td.select("a"):
        href = a.get("href", "")
        if "/sire/" in href or "sire_horse" in href:
            return "牡"
        if "/broodmare" in href or "/mare/" in href or "broodmare_horse" in href:
            return "牝"
    return ""


def extract_sires_from_ped_html(html: str) -> list[dict[str, Any]]:
    """
    blood_table HTML から sire 候補を抽出する。

    戻り値: [
        {
            "horse_id": str,       # netkeiba horse ID（空の場合あり）
            "name": str,           # 種牡馬名
            "sex": str,            # '牡' / '牝' / ''
            "sire_line": str,      # 「○○系」テキスト（取得できない場合は ""）
            "generation": int,     # 世代（1=父/母, 2=祖父/祖母, ...）
            "position": int,       # 世代内の位置
        },
        ...
    ]
    対象は全祖先。gen=1, pos=0 が直父（父）。
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(html, "html.parser")
    bt = soup.select_one(
        "table.blood_table, table[class*='blood'], table[summary*='血統']"
    )
    if not bt:
        return []

    results: list[dict[str, Any]] = []
    gen_counters: dict[int, int] = defaultdict(int)

    for td in bt.select("td"):
        rs = int(td.get("rowspan", 1))
        gen = _GEN_ROWSPAN.get(rs)
        if gen is None:
            continue

        pos = gen_counters[gen]
        gen_counters[gen] += 1

        link = td.select_one("a")
        if not link:
            continue
        name = link.get_text(strip=True)
        if not name:
            continue

        href = link.get("href", "")
        # /horse/{id}/ または /horse/sire/{id}/ の両パターンに対応
        # "sire" / "ped" 等のパス要素は除外し、8文字以上の英数字を ID として採用
        m = re.search(r"/horse/(?:sire/|ped/|broodmare/)?(\w{8,})", href)
        horse_id = m.group(1) if m else ""

        sire_line = _td_sire_line(td)
        sex = _td_sex(td)

        results.append(
            {
                "horse_id": horse_id,
                "name": name,
                "sex": sex,
                "sire_line": sire_line,
                "generation": gen,
                "position": pos,
            }
        )

    return results


def extract_direct_sire(html: str) -> dict[str, Any] | None:
    """血統 HTML から直父（generation=1, position=0）のみを返す。"""
    for anc in extract_sires_from_ped_html(html):
        if anc["generation"] == 1 and anc["position"] == 0:
            return anc
    return None
