"""クラス別の基準めぐ指数（突破ライン）。『○歳XXX』のクラスごとに、そのクラスの勝ち馬の指数の分布から作った目安。

- 指数は絶対指数（megu_v3a_abs_robust。1勝クラス・3歳以上を基準にした単一基準で、クラス間で比較できる）。
- 基準の表は config/class_benchmark.json（src/scripts/data/class_benchmark_build.py が作る）。
- 突破ライン = そのクラスの勝ち馬の指数の下位25%。これ以上なら、勝ち馬の4分の3以上と同水準。
- 勝ち馬の分布から作った目安で、突破を保証するものではない。
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
CONFIG_PATH = (_here.parents[3] if len(_here.parents) > 3 else _here.parent) / "config" / "class_benchmark.json"

_FULL = str.maketrans("０１２３４５６７８９", "0123456789")
_AGE = re.compile(r"([234])歳(以上)?")
_WINS = re.compile(r"([123])勝")

DIST_BANDS = ("短", "中短", "中", "長")


def _norm(s: Any) -> str:
    if s is None or s != s:
        return ""
    return re.sub(r"\s+", "", str(s).translate(_FULL))


def parse_age(*texts: Any) -> str | None:
    """『3歳』『3歳以上』『4歳以上』『2歳』。読めなければ None。"""
    for t in texts:
        m = _AGE.search(_norm(t))
        if m:
            return f"{m.group(1)}歳{'以上' if m.group(2) else ''}"
    return None


def parse_tier(race_class: Any, grade: Any, race_name: Any) -> str | None:
    """新馬 / 未勝利 / 1勝 / 2勝 / 3勝 / OP / L / G3 / G2 / G1。障害・読めないものは None。"""
    rc, gr, rn = _norm(race_class), _norm(grade), _norm(race_name)
    if "障害" in rc or "障害" in rn:
        return None
    g = gr.upper().replace("Ⅰ", "I")
    if g in ("G1", "GI"):
        return "G1"
    if g in ("G2", "GII"):
        return "G2"
    if g in ("G3", "GIII"):
        return "G3"
    if g in ("L", "LISTED"):
        return "L"
    for t in (rc, gr, rn):
        if "新馬" in t:
            return "新馬"
        if "未勝利" in t:
            return "未勝利"
    for t in (rc, gr):
        m = _WINS.search(t)
        if m:
            return f"{m.group(1)}勝"
    if g == "OP" or "オープン" in rc or "オープン" in gr:
        return "OP"
    m = _WINS.search(rn)           # 『ストークステークス(3勝)』のように、クラスがレース名にだけある
    if m:
        return f"{m.group(1)}勝"
    return None


def cell_of(race_class: Any, grade: Any, race_name: Any) -> tuple[str | None, str | None]:
    """(年齢, クラス)。クラスが読めなければ (None, None)。年齢は race_class → race_name の順に読む（無ければ None）。"""
    tier = parse_tier(race_class, grade, race_name)
    if tier is None:
        return None, None
    return parse_age(race_class, race_name), tier


def label_of(age: str | None, tier: str) -> str:
    name = {"1勝": "1勝クラス", "2勝": "2勝クラス", "3勝": "3勝クラス", "OP": "オープン"}.get(tier, tier)
    return f"{age or ''}{name}"


def surface_key(surface: Any) -> str | None:
    s = _norm(surface)
    if s.startswith("芝"):
        return "芝"
    if s.startswith("ダ"):
        return "ダ"
    return None


def dist_band(distance: Any) -> str | None:
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return None
    if d != d or d <= 0:
        return None
    return "短" if d <= 1300 else "中短" if d <= 1600 else "中" if d <= 2000 else "長"


def keys_for(age: str | None, tier: str, surface: str | None, band: str | None) -> list[str]:
    """表のキー。具体的なものから順（年齢×クラス×面×距離帯 → 年齢×クラス×面 → 年齢×クラス → クラスのみ）。"""
    a = age or "*"
    out: list[str] = []
    if surface and band:
        out.append(f"{a}|{tier}|{surface}|{band}")
    if surface:
        out.append(f"{a}|{tier}|{surface}")
    out.append(f"{a}|{tier}")
    if a != "*":
        out.append(f"*|{tier}")
    return out


_lock = threading.Lock()
_cache: dict[str, Any] = {}


def load_table(path: Path | None = None) -> dict | None:
    p = path or CONFIG_PATH
    if not p.exists():
        return None
    try:
        mt = p.stat().st_mtime
        with _lock:
            hit = _cache.get(str(p))
            if hit and hit[0] == mt:
                return hit[1]
        doc = json.loads(p.read_text(encoding="utf-8"))
        with _lock:
            _cache[str(p)] = (mt, doc)
        return doc
    except Exception:
        return None


def lookup(race_class: Any, grade: Any, race_name: Any, surface: Any, distance: Any, table: dict | None = None) -> dict | None:
    """そのレースの基準。クラスが読めない・表に無い場合は None（画面には何も出さない）。"""
    doc = table if table is not None else load_table()
    if not doc or not doc.get("cells"):
        return None
    age, tier = cell_of(race_class, grade, race_name)
    if tier is None:
        return None
    sf, band = surface_key(surface), dist_band(distance)
    for k in keys_for(age, tier, sf, band):
        c = doc["cells"].get(k)
        if c:
            parts = k.split("|")
            return {
                "label": label_of(age, tier), "key": k, "level": c.get("level") or ("cell" if len(parts) == 2 else "surface" if len(parts) == 3 else "distance"),
                "surface": parts[2] if len(parts) > 2 else None, "band": parts[3] if len(parts) > 3 else None,
                "n": c["n"], "median": c["median"], "line": c["p25"], "p75": c["p75"], "period": doc.get("period"),
                "fallback": k.split("|")[0] != (age or "*"),
            }
    return None
