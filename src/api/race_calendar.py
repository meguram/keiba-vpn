"""過去の開催日カレンダー（/races）用。月単位で『開催日・開催場・重賞』を返す純粋関数。"""

from __future__ import annotations

import re
from typing import Any, Callable

# netkeiba トップの Icon_GradeType<N>。1〜3 が平地の G1〜G3、10〜12 が障害の J.G1〜J.G3（東京ハイジャンプ=11 で確認）
_ICON_GRADE = {"1": "G1", "2": "G2", "3": "G3", "10": "J.G1", "11": "J.G2", "12": "J.G3"}
_NAME_GRADE = re.compile(r"[(（]\s*(J\.?\s*)?G\s*([123])\s*[)）]")


def grade_of(race: dict[str, Any]) -> str | None:
    """重賞なら 'G1'/'G2'/'G3'/'J.G1'…、そうでなければ None。アイコンを優先し、無い古い一覧はレース名の『(G3)』で拾う。"""
    icon = str(race.get("list_grade_icon") or "")
    m = re.fullmatch(r"Icon_GradeType(\d+)", icon)
    if m:
        return _ICON_GRADE.get(m.group(1))
    m = _NAME_GRADE.search(str(race.get("race_name") or ""))
    if m:
        return f"{'J.' if m.group(1) else ''}G{m.group(2)}"
    return None


def _short_name(name: str) -> str:
    name = re.sub(r"^第\s*\d+回\s*", "", name)
    return _NAME_GRADE.sub("", name).strip()


def day_summary(stem: str, races: list[dict[str, Any]]) -> dict[str, Any]:
    code = {}
    for r in races:
        v = r.get("venue")
        if v and v not in code:
            code[v] = str(r.get("race_id", ""))[4:6]
    grades = []
    for r in races:
        g = grade_of(r)
        if g:
            grades.append({
                "race_id": r.get("race_id"), "grade": g, "name": _short_name(str(r.get("race_name") or "")),
                "venue": r.get("venue") or "", "round": r.get("round"),
            })
    grades.sort(key=lambda x: (str(x["race_id"])[4:6], int(x["round"] or 0)))
    return {"date": stem, "n": len(races), "venues": sorted(code, key=lambda v: code[v]), "grades": grades}


def build_month(
    month: str,
    stems: list[str],
    load: Callable[[str], dict | None],
    is_race_day: Callable[[str, list, dict | None], bool],
    today: str,
) -> list[dict[str, Any]]:
    """month=YYYYMM。today（YYYYMMDD）より後の日は含めない。load(stem) は race_lists の中身。"""
    out = []
    for stem in sorted(s for s in stems if len(s) == 8 and s.isdigit() and s.startswith(month) and s <= today):
        rl = load(stem)
        if not rl:
            continue
        races = rl.get("races") or []
        if not is_race_day(stem, races, rl.get("_meta")):
            continue
        out.append(day_summary(stem, races))
    return out


def stem_range(stems: list[str], today: str) -> dict[str, str | None]:
    valid = sorted(s for s in stems if len(s) == 8 and s.isdigit() and s <= today)
    return {"min": valid[0] if valid else None, "max": valid[-1] if valid else None}
