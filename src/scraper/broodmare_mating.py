"""母馬の種付け情報（own.netkeiba.com 繁殖牝馬ページ）の取得・パース・産駒とのマッチング。

ページ: https://own.netkeiba.com/db/db_broodmare_record.html?id={母馬のhorse_id}
  表「種付け年 / 日付(月日) / 種牡馬名」。種牡馬ごとの**最終種付け日**で、データは2023年以降のみ。

産駒との対応: 種付け年 = 産駒の生まれ年 - 1。産駒の生まれ年は horse_id の先頭4桁。
保存形式の日付は ``yyyy-mm-dd``（種付け年 + ページの月日）。
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any

BROODMARE_MATING_URL = "https://own.netkeiba.com/db/db_broodmare_record.html?id={horse_id}"
MIN_MATING_YEAR = 2023  # ページの注記「種付け情報データは2023年以降」
MIN_FOAL_BIRTH_YEAR = MIN_MATING_YEAR + 1

_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_MONTH_DAY_RE = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_HORSE_ID_RE = re.compile(r"[?&]id=(\w+)")


def foal_birth_year(horse_id: str) -> int | None:
    """netkeiba の horse_id 先頭4桁は生まれ年。"""
    m = re.match(r"^(\d{4})\d{6}$", str(horse_id or "").strip())
    return int(m.group(1)) if m else None


def is_mating_lookup_eligible(foal_id: str) -> bool:
    """種付け情報が存在しうる産駒（2023年以降の種付け = 2024年以降生まれ）か。"""
    by = foal_birth_year(foal_id)
    return by is not None and by >= MIN_FOAL_BIRTH_YEAR


def dam_id_from_ancestors(ancestors: list[dict]) -> str:
    """5世代血統の祖先リストから母馬（generation=1, position=1）の horse_id を返す。"""
    for a in ancestors or []:
        if a.get("generation") == 1 and a.get("position") == 1:
            return str(a.get("horse_id") or "")
    return ""


def sire_from_ancestors(ancestors: list[dict]) -> tuple[str, str]:
    """父（generation=1, position=0）の (horse_id, name)。"""
    for a in ancestors or []:
        if a.get("generation") == 1 and a.get("position") == 0:
            return str(a.get("horse_id") or ""), str(a.get("name") or "")
    return "", ""


def _to_iso(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def parse_broodmare_mating(html: str) -> list[dict[str, Any]]:
    """種付け情報の表を ``[{mating_year, mating_date, sire_name, sire_id}, ...]`` にする。

    ページの行は ``<th>`` の中に ``<td>`` が入る不正な入れ子になっており、パーサによって木が変わる。
    そのため行のテキスト・リンクから正規表現で取り出す（木の形に依存しない）。
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    table = None
    for t in soup.select("table.NkOwnersTable01"):
        head = t.get_text(" ", strip=True)[:60]
        if "種付け年" in head:
            table = t
            break
    if table is None:
        return []

    records: list[dict[str, Any]] = []
    for tr in table.select("tbody tr") or table.select("tr"):
        link = tr.select_one("a.HorseName, a[href*='horse.html']")
        text = " ".join(tr.get_text(" ", strip=True).split())
        ym = _YEAR_RE.search(text)
        md = _MONTH_DAY_RE.search(text)
        if not (ym and md and link):
            continue  # ヘッダ行・注記行など
        year = int(ym.group(1))
        iso = _to_iso(year, int(md.group(1)), int(md.group(2)))
        if iso is None:
            continue  # 存在しない日付（ページ側の誤記など）は保持しない
        m_id = _HORSE_ID_RE.search(link.get("href", ""))
        records.append(
            {
                "mating_year": year,
                "mating_date": iso,
                "sire_name": link.get_text(strip=True),
                "sire_id": m_id.group(1) if m_id else "",
            }
        )
    return records


def build_broodmare_mating_record(dam_id: str, records: list[dict], *, source: str = "own_netkeiba") -> dict:
    return {
        "horse_id": dam_id,
        "matings": records,
        "mating_count": len(records),
        "source": source,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def match_foal_mating(
    matings: list[dict],
    foal_id: str,
    *,
    sire_id: str = "",
    sire_name: str = "",
) -> dict[str, Any]:
    """産駒に対応する種付けを選ぶ（種付け年 = 生まれ年 - 1）。

    返り値: ``{"mating_date": "yyyy-mm-dd" | None, "mating_year": int | None, "mating_match": <理由>}``
      - ``matched``: 該当年の種付けが1件、または複数件のうち父と一致した1件
      - ``no_record``: 該当年の種付けが無い（データ範囲外・未登録）
      - ``sire_mismatch``: 該当年の種付けはあるが、産駒の父と種牡馬が一致しない（別の種牡馬のため採用しない）
      - ``ambiguous``: 該当年に複数件あり父でも絞れない
      - ``out_of_range``: 産駒の生まれ年が種付け情報の範囲外（2023年種付け以前）
    父の情報が無い場合（``sire_id``/``sire_name`` とも空）は、年だけで決まるときに採用する。
    """
    by = foal_birth_year(foal_id)
    if by is None or by < MIN_FOAL_BIRTH_YEAR:
        return {"mating_date": None, "mating_year": None, "mating_match": "out_of_range"}
    year = by - 1
    cands = [m for m in matings or [] if m.get("mating_year") == year and m.get("mating_date")]
    if not cands:
        return {"mating_date": None, "mating_year": year, "mating_match": "no_record"}

    def same_sire(m: dict) -> bool:
        if sire_id and m.get("sire_id"):
            return m["sire_id"] == sire_id
        return bool(sire_name) and m.get("sire_name") == sire_name

    known_sire = bool(sire_id or sire_name)
    if known_sire:
        hits = [m for m in cands if same_sire(m)]
        if len(hits) == 1:
            return {"mating_date": hits[0]["mating_date"], "mating_year": year, "mating_match": "matched"}
        if not hits:
            return {"mating_date": None, "mating_year": year, "mating_match": "sire_mismatch"}
        return {"mating_date": None, "mating_year": year, "mating_match": "ambiguous"}
    if len(cands) == 1:
        return {"mating_date": cands[0]["mating_date"], "mating_year": year, "mating_match": "matched"}
    return {"mating_date": None, "mating_year": year, "mating_match": "ambiguous"}
