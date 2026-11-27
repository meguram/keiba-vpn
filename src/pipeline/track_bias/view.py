"""レース結果の API（/api/race/{id}/result）に足すトラックバイアスの塊。ホストで作った値（serving.py）を読むだけで、表示のときに計算しない。

読み出しは race_level と同じ（src/pipeline/race_time/levels.py の _read: 手元のファイル・無い／古いときはミラーの GCS から 1 件だけ取る・mtime のキャッシュ）。
費用（CLAUDE.md の最悪の回数）: 1 台あたり、meta 1 件と、開かれた年ごとに races・days の 2 件を、起動と 6 時間ごとの取り直しで読む。
最悪（全 11 年が開かれる）起動 96 回/日 × 23 件 ＋ 取り直し 4 回/日 × 23 件 ≒ 2,300 回/日 × 30 日 ≒ 69,000 回/月
（Class B 約 0.06 円/千回 → 約 4 円/月。race_level の分は levels.py で別に数える）。GCS に無いときは 1 時間に 1 回まで。
12/1 以降に gcs_mirror の TREES（束）に入れた後は、束で届くので、ここの取りは起動直後の 1 回だけになる。東京の中の読み出しなので転送料は無い。
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from src.pipeline.track_bias.serving import REL

NOTE = "その日の結果の振り返り。1 日の値の大半は偶然のばらつきで、次のレースの予測ではない。"
NOT_YET = "算出前（その日の結果の後に出ます）"
NOT_COVERED = "算出なし（障害・頭数 5 未満・オッズや通過順の欠けのレースは対象外）"
NOT_PLACED = "算出なし（この年の値はまだ置いていません）"
FORMULA = ("市場の見込み（確定単勝オッズからの 3 着内の確率）に比べて、序盤に前にいた馬・内にいた馬がどれだけ来たかを、"
           "同じ場・面・距離・クラス・馬場のふだんの日と比べた値。z は ±1 で有利、±2 で強い。位置の番手は 番手/頭数。")


def _f(v, nd=2):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def view_of(row: dict[str, Any] | None, day: dict[str, Any] | None, meta: dict[str, Any], date: str = "") -> dict[str, Any]:
    if not row or not day:
        last = str(meta.get("last_date") or "")
        covered = bool(last) and str(date)[:10] <= last          # 作った日までのレースで値が無いのは、対象外（ずっと出ない）
        return {"available": False, "note": NOT_COVERED if covered else NOT_YET, "version": meta.get("version")}
    try:
        top3 = json.loads(row.get("top3") or "[]")
    except (TypeError, ValueError):
        top3 = []
    return {
        "available": True, "version": meta.get("version"), "built": meta.get("built"), "note": NOTE, "formula": FORMULA,
        "day": {"venue": day.get("venue"), "surf": day.get("surf"), "label": day.get("label"), "front_label": day.get("front_label"),
                "inner_label": day.get("inner_label"), "z_front": _f(day.get("z_front"), 1), "z_inner": _f(day.get("z_inner"), 1),
                "n_races": None if day.get("n_races") is None else int(day["n_races"])},
        "race": {"front_se": _f(row.get("front_se")), "front_zc": _f(row.get("front_zc")), "inner_se": _f(row.get("inner_se")),
                 "inner_zc": _f(row.get("inner_zc")), "front4c_se": _f(row.get("front4c_se"))},
        "top3": top3,
    }


def _iso(date: str) -> str:
    d = str(date or "").replace("-", "")
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) >= 8 and d[:8].isdigit() else str(date or "")


def result_block(race_id: str, date: str) -> dict[str, Any] | None:
    """値が無ければ available=False（作った日より後は「算出前」、作った日までは「算出なし」）。
    meta.json がまだ置かれていない（ホストの作りもミラーの置き場も無い）ときは None（API に塊を付けない）。"""
    from src.pipeline.race_time.levels import _read

    date = _iso(date)
    y = date[:4]
    meta = _read(f"{REL}/meta.json", "json") or {}
    if not meta:
        return None
    races = _read(f"{REL}/races_{y}.parquet", "parquet") if y else None
    if races is None or races.empty:
        return {"available": False, "note": NOT_PLACED, "version": meta.get("version")}
    hit = races[races["race_id"].astype(str) == str(race_id)]
    if hit.empty:
        return view_of(None, None, meta, date)
    row = hit.iloc[0].to_dict()
    days = _read(f"{REL}/days_{y}.parquet", "parquet")
    day = None
    if days is not None and not days.empty:
        dh = days[(days["date"] == row["date"]) & (days["venue"] == row["venue"]) & (days["surf"] == row["surf"])]
        day = dh.iloc[0].to_dict() if not dh.empty else None
    return view_of(row, day, meta, date)
