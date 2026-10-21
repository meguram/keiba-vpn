"""馬プロフィール表の追加項目（セリ取引価格・馬主 ID・近親馬）のパース。表の見出しと本文・リンク先だけを入力にする（HTML 非依存）。

セリ取引価格の書式（実測）: 「1,836万円 (2014年 北海道セレクションセール)」、無いときは「-」。
馬主・生産者・近親馬はリンク先（/owner/<ID>/, /breeder/<ID>/, /horse/<ID>/）から ID を取る。ID は数字とは限らない（例 x0a92b）。
"""

from __future__ import annotations

import re
from typing import Any

_PRICE = re.compile(r"(?:([\d,\.]+)\s*億)?\s*(?:([\d,\.]+)\s*万)?\s*円")
_YEAR_SALE = re.compile(r"\(\s*(\d{4})年\s*([^)]*)\)")


def parse_sale_price(text: str) -> dict[str, Any]:
    """セリ取引価格の文字列 → {price_man, sale_year, sale_name}。価格が無い・読めないときは全て None（読めない文字列は raw に残す）。"""
    t = (text or "").strip()
    out: dict[str, Any] = {"price_man": None, "sale_year": None, "sale_name": None}
    if not t or t in {"-", "－", "―"}:
        return out
    m = _PRICE.search(t.replace("，", ","))
    if m and (m.group(1) or m.group(2)):
        try:
            out["price_man"] = float((m.group(1) or "0").replace(",", "")) * 10000 + float((m.group(2) or "0").replace(",", ""))
        except ValueError:
            pass
    y = _YEAR_SALE.search(t)
    if y:
        out["sale_year"] = int(y.group(1))
        out["sale_name"] = y.group(2).strip() or None
    if out["price_man"] is None:
        out["raw"] = t[:60]
    return out


def id_from_href(href: str, kind: str) -> str:
    m = re.search(rf"/{kind}/([^/?#]+)", str(href or ""))
    return m.group(1) if m else ""


def parse_profile_rows(rows: list[dict]) -> dict[str, Any]:
    """プロフィール表の行（th / td / hrefs）→ 追加項目。rows は acq_profile_pilot.extract_rows の rows。"""
    by = {r["th"]: r for r in rows}
    out: dict[str, Any] = {}
    if "セリ取引価格" in by:
        p = parse_sale_price(by["セリ取引価格"]["td"])
        out.update({"sale_price_man": p["price_man"], "sale_year": p["sale_year"], "sale_name": p["sale_name"], "sale_raw": p.get("raw")})
        out["sale_listed"] = 1
    else:
        out["sale_listed"] = 0
    for th, kind in (("馬主", "owner"), ("生産者", "breeder")):
        r = by.get(th)
        hrefs = (r or {}).get("hrefs") or []
        out[f"{kind}_id"] = id_from_href(hrefs[0], kind) if hrefs else ""
    r = by.get("近親馬")
    out["relatives"] = [id_from_href(h, "horse") for h in (r or {}).get("hrefs") or [] if id_from_href(h, "horse")]
    return out
