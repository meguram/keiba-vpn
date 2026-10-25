"""推奨馬券を正として、印（◎○▲△）を買い目から決め直す。

推奨馬券（target_bets の既定プラン）を絶対とし、めぐ印はその付随情報にする。各馬の購入額（組合せの購入額を脚の数で割る。
馬単・3連単は1着側を2倍に重み付け）の合計が大きい順に ◎ ○ ▲、買い目に入った残りの馬は △。買い目に入っていない馬には印を付けない
（消印だけは、買わない馬への注意なので残す）。買い目が無いレース（見送り）は、従来の印のままにする。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

ORDERED_TYPES = ("umatan", "sanrentan")
_TOP = ("honmei", "pair", "show_val")
SYMBOLS = {"honmei": "◎", "pair": "○", "show_val": "▲", "anchor": "△", "star": "★", "kesi": "消"}
LABELS = {
    "honmei": "◎ 買い目の軸（購入額が最大）", "pair": "○ 買い目の2番手", "show_val": "▲ 買い目の3番手", "anchor": "△ 買い目に入る馬", "none": "",
}


def horse_stakes(bets: Iterable[dict[str, Any]]) -> dict[int, float]:
    """馬番 → 買い目での購入額の合計（円）。"""
    out: dict[int, float] = defaultdict(float)
    for b in bets or []:
        combo = [int(h) for h in (b.get("combo") or [])]
        yen = float(b.get("yen") or 0)
        if not combo or yen <= 0:
            continue
        pos = [2.0] + [1.0] * (len(combo) - 1) if b.get("bet_type") in ORDERED_TYPES else [1.0] * len(combo)
        tot = sum(pos)
        for h, p in zip(combo, pos):
            out[h] += yen * p / tot
    return dict(out)


def aligned_mark_types(entries: list[dict[str, Any]], bets: Iterable[dict[str, Any]] | None) -> dict[int, str] | None:
    """買い目から決めた 馬番 → mark_type。買い目が無ければ None（従来の印のまま）。"""
    w = horse_stakes(bets or [])
    if not w:
        return None
    win = {int(e.get("horse_number") or 0): float(e.get("win_prob") or 0.0) for e in entries}
    order = sorted(w, key=lambda h: (-round(w[h], 6), -win.get(h, 0.0), h))
    return {h: (_TOP[i] if i < len(_TOP) else "anchor") for i, h in enumerate(order)}


def apply_to_entries(entries: list[dict[str, Any]], marks: dict[int, str]) -> list[dict[str, Any]]:
    """出走馬の一覧に、買い目からの印を付けた新しい一覧を返す。買い目に無い馬は印なし（消印だけ残す）。"""
    out = []
    for e in entries:
        if not isinstance(e, dict):
            out.append(e)
            continue
        e = dict(e)
        mt = marks.get(int(e.get("horse_number") or 0))
        if mt is None:
            mt = "kesi" if e.get("mark_type") == "kesi" else "none"
        e["mark_type"] = mt
        if mt != "kesi":
            e["recommendation"] = LABELS.get(mt, "")
        out.append(e)
    return out


def relabel_legs(portfolio: dict[str, Any], marks: dict[int, str]) -> None:
    """推奨ポートフォリオ（既定プランと全プラン）の各買い目の脚の印を、買い目からの印に揃える（in-place）。"""
    lists = [portfolio.get("bets") or []]
    for pl in portfolio.get("plans") or []:
        lists.append(pl.get("bets") or [])
    for bets in lists:
        for b in bets:
            for leg in b.get("legs") or []:
                leg["mark"] = SYMBOLS.get(marks.get(int(leg.get("horse_number") or 0), ""), "")
