"""種牡馬の系統（父系）の分類。馬の血統（5代）の父系の祖先を父から順にさかのぼり、最初に当たる『系統の祖』でまとめる。

サンプル数が少ない種牡馬を、同じ系統の産駒としてまとめて見るために使う。系統の祖は馬 ID（netkeiba）で持ち、名前では結合しない。
どれにも当たらない馬も『その他』にはまとめず、わかる範囲でいちばん古い父系の祖の系統にする。
"""

from __future__ import annotations

import re
from typing import Any, Callable

# 系統の祖の馬 ID -> 表示名。近い祖ほど優先されるので、孫系統（キングカメハメハ・デインヒル等）は祖より先に当たる。
LINES: dict[str, str] = {
    "000a00033a": "サンデーサイレンス系",
    "2001103460": "キングカメハメハ系",
    "000a001d7e": "キングマンボ系",
    "000a001607": "ミスタープロスペクター系",
    "000a0000d1": "デインヒル系",
    "000a0000d3": "ダンチヒ系",
    "000a001a98": "ストームキャット系",
    "000a00185d": "サドラーズウェルズ系",
    "000a000e04": "ノーザンダンサー系",
    "000a0012bf": "ヘイロー系",
    "000a0012cb": "ロベルト系",
    "000a000f2b": "ヘイルトゥリーズン系",
    "000a001d37": "A.P.インディ系",
    "000a0015fc": "シアトルスルー系",
    "000a000355": "テスコボーイ系",
    "000a000f88": "ナスルーラ系",
    "000a000f8c": "ニアークティック系",
}


def _clean_name(name: str) -> str:
    """『サンデーサイレンスSunday Silence(米)』→『サンデーサイレンス』、『Hail to Reason』→『Hail to Reason』。"""
    name = (name or "").replace("\xa0", " ").strip()
    m = re.match(r"^[^A-Za-z(（]+", name)
    return (m.group(0) if m else re.sub(r"[(（].*$", "", name)).strip()


def _chain(ancestors: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """父系（位置0）の祖を、近い順（父・父の父…）に [(馬ID, 名前)]。"""
    by_gen: dict[int, tuple[str, str]] = {}
    for a in ancestors or []:
        try:
            if int(a["position"]) == 0 and a.get("horse_id"):
                by_gen[int(a["generation"])] = (str(a["horse_id"]), str(a.get("name") or ""))
        except (KeyError, TypeError, ValueError):
            continue
    return [by_gen[g] for g in sorted(by_gen)]


def line_of(ancestors: list[dict[str, Any]], lookup: Callable[[str], list[dict[str, Any]] | None] | None = None, max_len: int = 7) -> tuple[str, str] | None:
    """血統 JSON の ancestors から、父の系統 (馬ID, 表示名) を返す。
    父から父の父…（位置0）の順に、上の LINES に最初に当たった祖。5代で当たらなければ、いちばん古い祖自身の血統（lookup(馬ID) -> ancestors）を引いてさらにさかのぼる。
    それでも（7代まで）当たらなければ『その他』にはまとめず、わかる範囲でいちばん古い父系の祖そのものの系統にする。"""
    chain = _chain(ancestors)
    if not chain:
        return None
    seen = {h for h, _ in chain}
    while True:
        for hid, _name in chain[:max_len]:
            if hid in LINES:
                return hid, LINES[hid]
        if lookup is None or len(chain) >= max_len:
            break
        more = [c for c in _chain(lookup(chain[-1][0]) or []) if c[0] not in seen]
        if not more:
            break
        seen |= {h for h, _ in more}
        chain = chain + more
    hid, name = chain[min(len(chain), max_len) - 1]
    nm = _clean_name(name)
    return (hid, f"{nm}系") if nm else (hid, hid)
