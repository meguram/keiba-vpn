"""JRA コースの形状データ（直線の長さ・コーナー数・スタート位置・坂 など）を、独立した複数の情報源から突き合わせて検証する。

方針: 確認できた値だけを使う。複数の情報源が許容誤差内で一致した値を「合意」、一つの情報源だけの値を「未確認」、食い違う値を「不一致」とし、
不一致は人が出典を確認して解消する（このモジュールは解消しない）。データ（実レース）から測れる量（コーナー数、200m ごとのラップの形）とも照合する。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

NUMERIC_TOL = {"lap_length_m": 30.0, "final_straight_m": 10.0, "dist_start_to_first_corner_m": 40.0, "elevation_range_m": 0.4, "n_corners": 0.0, "n_laps": 0.0}
KEY_FIELDS = ["venue_code", "surface", "distance"]


def to_num(x: Any) -> float | None:
    """「約 525m」「1,624.1」「2.4%」などから最初の数値を取り出す。数値が無ければ None。"""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"-?\d[\d,]*\.?\d*", str(x))
    return float(m.group(0).replace(",", "")) if m else None


def canon_surface(s: Any) -> str:
    s = str(s or "")
    return "芝" if "芝" in s else ("ダート" if "ダ" in s else s)


def canon_variant(v: Any) -> str:
    v = str(v or "")
    if "内" in v:
        return "内"
    if "外" in v:
        return "外"
    if "直" in v:
        return "直"
    return ""


def row_key(r: dict) -> tuple:
    return (str(r.get("venue_code") or "").zfill(2), canon_surface(r.get("surface")), int(to_num(r.get("distance")) or 0), canon_variant(r.get("variant")))


def has_goal_slope(text: Any) -> bool | None:
    """goal_slope の自由記述から、ゴール前に坂があるかを読む。「なし」「平坦」は False、坂・勾配・上りの記述は True。判断できなければ None。"""
    if text is None:
        return None
    t = json.dumps(text, ensure_ascii=False) if not isinstance(text, str) else text
    if not t.strip() or t.strip().lower() in ("null", "none"):
        return None
    if re.search(r"(坂|勾配|上り|登り|昇り|%)", t) and not re.search(r"^(なし|無し|ほぼ平坦|平坦)", t.strip()):
        return not re.search(r"(坂なし|坂は無い|坂はない|平坦|勾配なし|ほとんど無)", t) or bool(re.search(r"(急坂|上り坂|登り坂|\d\s*%|\d\.\d\s*m)", t))
    return False if re.search(r"(なし|無し|平坦)", t) else None


def load(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(sources: dict[str, list[dict]], fields=tuple(NUMERIC_TOL)) -> pd.DataFrame:
    """長い形式: コース（key）× 項目 × 情報源の値 と、合意の判定。
    判定: 合意 = 2 つ以上の情報源が許容誤差内で一致し、外れ値が無い / 一致なし = 値を持つ情報源が 2 つ以上あり許容内に収まらない / 未確認 = 1 つの情報源だけ / 欠損 = 全て null"""
    idx: dict[tuple, dict[str, dict]] = {}
    for name, rows in sources.items():
        for r in rows:
            idx.setdefault(row_key(r), {})[name] = r
    out = []
    for key, by in sorted(idx.items()):
        for f in fields:
            vals = {n: to_num(r.get(f)) for n, r in by.items()}
            have = {n: v for n, v in vals.items() if v is not None}
            tol = NUMERIC_TOL.get(f, 0.0)
            if not have:
                status, cons = "欠損", None
            elif len(have) == 1:
                status, cons = "未確認", next(iter(have.values()))
            else:
                med = median(have.values())
                ok = [n for n, v in have.items() if abs(v - med) <= tol]
                if len(ok) == len(have):
                    status, cons = "合意", median(have.values())
                elif len(ok) >= 2:
                    status, cons = "合意(外れ値あり)", median([have[n] for n in ok])
                else:
                    status, cons = "不一致", None
            out.append({"venue_code": key[0], "surface": key[1], "distance": key[2], "variant": key[3], "field": f, "status": status, "consensus": cons, **{f"v_{n}": vals.get(n) for n in sources}})
    return pd.DataFrame(out)


def status_summary(cmp_: pd.DataFrame) -> pd.DataFrame:
    return cmp_.pivot_table(index="field", columns="status", values="distance", aggfunc="count", fill_value=0)
