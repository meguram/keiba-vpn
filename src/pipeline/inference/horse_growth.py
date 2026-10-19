"""馬の成長曲線（過去走ごとの『めぐ指数』の推移）を、保存済みのデータから組み立てる。

縦軸はユーザー独自の『めぐ指数』（``data/page_reference/tables/<年>/megu_index_flat.parquet``、馬×レースごと、2020 年〜）。
ただし年・時期によって水準が段差状にずれている（ホストの 2024 年は月中央値が約 45、2025 年は約 94、2026-05 以降は約 55）。
そのまま並べると、年をまたいだ曲線が偽の上昇・下降になるので、『その月の全馬の中央値』を引いて全期間の基準値に揃えた値を縦軸にし、
元の値（raw）も各点に残す。係数を同じ版で作り直した後は補正が不要になる（補正量が 0 に近づく）。

対象レース日より前の走だけを使う（呼び出し側 entry_card が all_runs を日付で絞る）。重い計算はしない（馬 ID で絞って読むだけ）。
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
FIRST_YEAR = 2020
TREND_FLAT = 3.0          # 直近と、その前の平均の差がこの指数以内なら『横ばい』
TREND_WINDOW = 3


@dataclass(frozen=True)
class Axis:
    """縦軸にする指数の定義。指数を差し替えるときは AXES に 1 つ足し、KEIBA_GROWTH_AXIS で選ぶ（画面には name がそのまま出る）。"""
    key: str
    name: str                 # 画面に出す指数名
    file: str                 # data/page_reference/tables/<年>/ 直下の parquet（馬×レース単位。race_id・horse_id を持つ）
    value_col: str
    status_col: str | None    # None なら全行を有効とみなす
    valid: str | tuple[str, ...] = "valid"   # 有効とみなす状態（複数可）
    align_by_month: bool = True   # 年・時期の水準のずれを、月ごとの全馬の中央値で揃える
    conf_col: str | None = None   # True の行は『信頼度が低い』として点に印を付ける（値は使う）

    @property
    def valid_set(self) -> frozenset[str]:
        return frozenset([self.valid] if isinstance(self.valid, str) else self.valid)


AXES: dict[str, Axis] = {
    "megu": Axis("megu", "めぐ指数", "megu_index_flat.parquet", "megu_index", "computation_status"),
    # 全期間を同じ版で作り直した列（範囲外の復元・[0,100] 丸め済み）。水準がそろっているので月別補正は不要
    "megu_robust": Axis(
        "megu_robust", "めぐ指数", "megu_index_flat.parquet", "megu_index_robust", "megu_index_status",
        valid=("ok", "clipped", "restored_out_of_range", "restored_clipped"), align_by_month=False,
        conf_col="low_confidence",
    ),
}
DEFAULT_AXIS = "megu_robust"


def active_axis() -> Axis:
    return AXES.get(os.environ.get("KEIBA_GROWTH_AXIS") or DEFAULT_AXIS) or AXES[DEFAULT_AXIS]


def tables_dir() -> Path:
    try:
        from src.config.data_paths import page_reference_root

        return page_reference_root() / "tables"
    except Exception:
        return ROOT / "data" / "page_reference" / "tables"


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


# ── 純粋関数（テスト対象）─────────────────────────────────────────────

def parse_birthday(s: Any) -> tuple[int, int, int] | None:
    """'2022年4月5日' / '2022/04/05' / '2022-04-05' / '20220405' → (年, 月, 日)。読めなければ None。"""
    t = str(s or "").strip()
    m = re.search(r"(\d{4})\D{0,2}(\d{1,2})\D{0,2}(\d{1,2})", t)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return (y, mo, d) if 1990 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31 else None


def age_months(birthday: tuple[int, int, int] | None, ymd: str) -> int | None:
    """生年月日から、ymd（YYYY-MM-DD / YYYYMMDD）時点の満月齢。"""
    if not birthday:
        return None
    m = re.match(r"^(\d{4})\D?(\d{2})\D?(\d{2})$", str(ymd or ""))
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    months = (y - birthday[0]) * 12 + (mo - birthday[1]) - (1 if d < birthday[2] else 0)
    return months if months >= 0 else None


def align(raw: float, month_median: float | None, ref: float) -> float:
    """その月の水準差を引いて全期間の基準値に揃える。月の中央値が無ければ補正しない。"""
    return raw if month_median is None else raw - month_median + ref


def trend(values: list[float]) -> dict | None:
    """直近 3 走の平均 − その前 3 走の平均（走数が少ない時は直近 − 初回）で、上昇中/横ばい/下降中。2 点未満は None。"""
    if len(values) < 2:
        return None
    if len(values) >= TREND_WINDOW + 1:
        recent = values[-TREND_WINDOW:]
        prev = values[-2 * TREND_WINDOW:-TREND_WINDOW] or values[:-TREND_WINDOW]
        basis = f"直近{len(recent)}走の平均 − その前{len(prev)}走の平均"
        diff = sum(recent) / len(recent) - sum(prev) / len(prev)
    else:
        diff, basis = values[-1] - values[0], "直近 − 初回"
    label = "上昇中" if diff > TREND_FLAT else "下降中" if diff < -TREND_FLAT else "横ばい"
    return {"label": label, "delta": round(diff, 1), "basis": basis}


def build_growth(
    runs: list[dict],
    megu: dict[str, tuple[float, str]],
    baseline: dict[str, float],
    ref: float,
    birthday: Any = None,
    axis: Axis | None = None,
) -> dict:
    """過去走（日付昇順に並べ替える）と、race_id → (めぐ指数, 状態) から成長曲線を作る。

    status: ok（1 点以上）/ no_data（めぐ指数が 1 つも無い）。点が 1 点だけの時もデータは返し、UI が『曲線は出せません』と出す。
    """
    ax = axis or active_axis()
    bd = parse_birthday(birthday)
    pts: list[dict] = []
    excluded = 0
    for r in sorted((x for x in runs if x.get("date")), key=lambda x: x["date"]):
        rid = str(r.get("race_id") or "")
        hit = megu.get(rid)
        if not hit:
            continue
        val, status = hit
        low_conf = status.endswith("+low")
        if low_conf:
            status = status[:-4]
        if status not in ax.valid_set or val is None:
            excluded += 1
            continue
        d8 = re.sub(r"\D", "", str(r["date"]))[:8]
        month = f"{d8[:4]}-{d8[4:6]}"
        pts.append({
            "date": d8, "race_id": rid, "race_name": r.get("race_name"), "venue": r.get("venue"),
            "surface": r.get("surface"), "distance": r.get("distance"), "finish": r.get("finish"), "field": r.get("field"),
            "age_months": age_months(bd, r["date"]), "raw": round(float(val), 1), "value": round(align(float(val), baseline.get(month) if ax.align_by_month else None, ref), 1),
            "low_confidence": low_conf,
        })
    if not pts:
        return {"status": "no_data", "reason": f"{ax.name}が未計算、または基準外です", "excluded": excluded}
    best = max(pts, key=lambda p: p["value"])
    return {
        "status": "ok", "axis": {"key": ax.key, "name": ax.name, "note": "時期ごとの水準差をそろえた値（元の値は各点に表示）" if ax.align_by_month else "元の値", "ref": round(ref, 1)},
        "points": pts, "n": len(pts), "excluded": excluded, "trend": trend([p["value"] for p in pts]),
        "best": {"value": best["value"], "date": best["date"], "race_name": best["race_name"]},
    }


# ── 保存済みデータの読み込み ─────────────────────────────────────────────

_lock = threading.Lock()
_baseline_cache: dict[str, Any] = {}


def _read(path: Path, **kw):
    import pandas as pd

    return pd.read_parquet(path, **kw)


def megu_map_for_horse(horse_id: str, years: set[int] | None = None, base: Path | None = None, axis: Axis | None = None) -> dict[str, tuple[float, str]]:
    """馬 ID で絞って、race_id → (指数, 計算状態)。ファイルが無い年は飛ばす。"""
    ax = axis or active_axis()
    base = base or tables_dir()
    out: dict[str, tuple[float, str]] = {}
    for y in sorted(years) if years else range(FIRST_YEAR, 2100):
        p = base / str(y) / ax.file
        if not p.exists():
            if years is None and y > FIRST_YEAR + 10:
                break
            continue
        try:
            cols = ["race_id", "horse_id", ax.value_col] + ([ax.status_col] if ax.status_col else []) + ([ax.conf_col] if ax.conf_col else [])
            df = _read(p, columns=cols, filters=[("horse_id", "==", horse_id)])
        except Exception:
            continue
        first_valid = sorted(ax.valid_set)[0]
        sts = df[ax.status_col] if ax.status_col else [first_valid] * len(df)
        lows = df[ax.conf_col] if ax.conf_col else [False] * len(df)
        for rid, v, st, lo in zip(df["race_id"], df[ax.value_col], sts, lows):
            if v == v:
                out[str(rid)] = (float(v), str(st) + ("+low" if lo is True or lo == 1 else ""))
    return out


def month_baseline(base: Path | None = None, axis: Axis | None = None) -> tuple[dict[str, float], float]:
    """月ごとの『有効な指数』の全馬の中央値と、全期間の基準値（月中央値の中央値）。ファイルの更新時刻でキャッシュ。"""
    ax = axis or active_axis()
    base = base or tables_dir()
    files = sorted(base.glob(f"*/{ax.file}"))
    sig = (ax.key,) + tuple((str(f), _mtime(f)) for f in files)
    with _lock:
        hit = _baseline_cache.get("v")
        if hit and hit[0] == sig:
            return hit[1], hit[2]
    import pandas as pd

    monthly: dict[str, list[float]] = {}
    for f in files:
        rr = f.parent / "race_result_flat.parquet"
        if not rr.exists():
            continue
        try:
            m = _read(f, columns=["race_id", ax.value_col] + ([ax.status_col] if ax.status_col else []))
            m = m[m[ax.value_col].notna()]
            if ax.status_col:
                m = m[m[ax.status_col].isin(ax.valid_set)]
            d = _read(rr, columns=["race_id", "date"]).drop_duplicates("race_id")
            j = m.merge(d, on="race_id")
            j["ym"] = j["date"].astype(str).str.replace("/", "-").str[:7]
            for ym, g in j.groupby("ym")[ax.value_col]:
                monthly.setdefault(ym, []).extend(g.tolist())
        except Exception:
            continue
    base_med = {ym: float(median(v)) for ym, v in monthly.items() if len(v) >= 200}
    ref = float(median(base_med.values())) if base_med else 50.0
    with _lock:
        _baseline_cache["v"] = (sig, base_med, ref)
    return base_med, ref


def growth_for_horse(horse_id: str, runs: list[dict], birthday: Any = None, base: Path | None = None) -> dict:
    """entry_card から呼ぶ入口。失敗しても例外を出さず status を返す。縦軸の指数は active_axis()（KEIBA_GROWTH_AXIS）。"""
    ax = active_axis()
    try:
        years = {int(str(r["date"])[:4]) for r in runs if r.get("date")}
        if not years:
            return {"status": "no_data", "reason": "出走履歴がありません"}
        megu = megu_map_for_horse(horse_id, years, base, ax)
        if not megu:
            return {"status": "no_data", "reason": f"この馬の{ax.name}がまだ計算されていません"}
        baseline, ref = month_baseline(base, ax)
        return build_growth(runs, megu, baseline, ref, birthday, ax)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reason": f"{type(exc).__name__}"}
