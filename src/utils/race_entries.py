"""出馬表 entries の馬番・枠番補完（欠落データ向け）。"""

from __future__ import annotations

from typing import Any


def _positive_int(val: Any) -> int:
    try:
        n = int(val or 0)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def normalize_race_entries(entries: list[dict]) -> list[dict]:
    """
    馬番・枠番が 0 または重複している出馬表を推論可能な形に補完する。

    - 全頭有効な馬番がユニークならそのまま（欠落枠のみ補完）
    - それ以外は出馬表順（人気順があれば人気昇順）で 1..N を割当
    """
    if not entries:
        return []

    out = [dict(e) for e in entries]
    horse_nums = [_positive_int(e.get("horse_number")) for e in out]
    n = len(out)
    valid = [h for h in horse_nums if h > 0]
    all_valid_unique = (
        len(valid) == n
        and len(set(valid)) == n
        and max(valid) <= max(n, max(valid))
    )

    if all_valid_unique:
        for e in out:
            hn = _positive_int(e.get("horse_number"))
            if _positive_int(e.get("bracket_number")) <= 0:
                e["bracket_number"] = max(1, (hn + 1) // 2)
        return out

    indexed = list(enumerate(out))
    if any(_positive_int(e.get("popularity")) > 0 for _, e in indexed):
        indexed.sort(
            key=lambda ie: (_positive_int(ie[1].get("popularity")), ie[0]),
        )

    used: set[int] = set()
    if len(valid) == len(set(valid)) and len(valid) >= n // 2:
        used = set(valid)

    seq = 1
    for _idx, e in indexed:
        hn = _positive_int(e.get("horse_number"))
        if hn <= 0 or hn in used or not all_valid_unique:
            while seq in used:
                seq += 1
            e["horse_number"] = seq
            e["_horse_number_inferred"] = 1
            used.add(seq)
            seq += 1
        else:
            used.add(hn)

        if _positive_int(e.get("bracket_number")) <= 0:
            hn_final = _positive_int(e.get("horse_number"))
            e["bracket_number"] = max(1, (hn_final + 1) // 2)

    out.sort(key=lambda e: _positive_int(e.get("horse_number")))
    return out
