"""レースレベル・開催日レベル・メンバーレベル（基準の走破時計の関数 v2 から。ホストで事前に計算して置き、表示は読むだけ）。

- レースレベル = 1〜5 着の走破時計の中央値 − T̂（秒。負なら基準より速い）。ペースの補正の前（ペースの項を 0）の値も並べる。
- 開催日レベル = その日・場・芝ダのレースレベル（1000m あたり）の中央値を、件数で 0 へ縮めた値（w = n/(n+k)）。
- メンバーレベル = レースレベル − 自分のレースを除いた開催日レベル（同じ日・場・芝ダのほかのレースから作る。自分のレースで馬場を測らない）。
- 言葉の段（1000m あたりの秒。数字を見る前に固定 2026-11-26）:
  馬場・レース: ≤−0.6 速い / −0.6〜−0.2 やや速い / ±0.2 標準 / 0.2〜0.6 やや時計が掛かる / ≥0.6 時計が掛かる
  メンバー: ≤−0.4 高い / −0.4〜−0.15 やや高い / ±0.15 標準 / 0.15〜0.4 やや低い / ≥0.4 低い
  ペース: 前半 600m の z（コースの中の標準化）≤−0.67 ハイ / ≥+0.67 スロー / それ以外 平均
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.race_time.model import RaceTimeModel

TRACK_WORDS = ((-0.6, "速い"), (-0.2, "やや速い"), (0.2, "標準"), (0.6, "やや時計が掛かる"), (float("inf"), "時計が掛かる"))
MEMBER_WORDS = ((-0.4, "高い"), (-0.15, "やや高い"), (0.15, "標準"), (0.4, "やや低い"), (float("inf"), "低い"))
PACE_Z = 0.67
K_MIN, K_MAX = 2.0, 10.0


def word(per1k: float | None, table=TRACK_WORDS) -> str | None:
    """1000m あたりの秒から言葉の段。境の値ちょうどは速い（高い）側に入れる。"""
    if per1k is None or not np.isfinite(per1k):
        return None
    for edge, w in table:
        if per1k <= edge:
            return w
    return table[-1][1]


def pace_label(z: float | None) -> str | None:
    if z is None or not np.isfinite(z):
        return None
    return "ハイ" if z <= -PACE_Z else "スロー" if z >= PACE_Z else "平均"


def shrink_k(per1k: pd.Series, keys: pd.DataFrame, clip: bool = True) -> float:
    """縮みの強さ k = 日の中のばらつき ÷ 日の間のばらつき（clip=True なら 2〜10 に切る）。学習の期間の値で決めて固定する。
    近い形の推定（平均の件数・平均の分散）で、開催日レベルは中央値なので本来の k はもう少し大きい（12/1 以降に直す）。"""
    d = pd.concat([keys.reset_index(drop=True), per1k.reset_index(drop=True).rename("v")], axis=1).dropna()
    g = d.groupby(["date", "venue", "surf"])["v"]
    within = float(g.var(ddof=1).dropna().mean())
    between = float(g.mean().var(ddof=1)) - within / float(g.size().mean())
    if not np.isfinite(within) or not np.isfinite(between) or between <= 0:
        return K_MAX if clip else float("nan")
    return float(np.clip(within / between, K_MIN, K_MAX)) if clip else float(within / between)


def _shrunk_median(v: np.ndarray, k: float) -> float:
    n = len(v)
    return float(np.median(v) * n / (n + k)) if n else 0.0


def compute(races: pd.DataFrame, model: RaceTimeModel, k: float, win_model: RaceTimeModel | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """レースの表（race_table の行）から、レースの行と日の行を作る。"""
    d = races.dropna(subset=["front600", "y_top5"]).reset_index(drop=True).copy()
    parts = model.predict(d, parts=True)
    d["t_hat"] = parts["full"]
    d["race_level"] = d["y_top5"] - parts["full"]
    d["race_level_raw"] = d["y_top5"] - parts["no_pace"]
    km = d["distance"] / 1000.0
    d["race_level_per1k"] = d["race_level"] / km
    d["race_level_raw_per1k"] = d["race_level_raw"] / km
    ref = d["course"].map(model.pace_ref)
    sd = d["course"].map(model.pace_sd)
    d["pace_z"] = (d["front600"] - ref) / sd
    d["pace_label"] = d["pace_z"].map(pace_label)
    if win_model is not None:
        d["race_level_win"] = d["y_win"] - win_model.predict(d)
        d["race_level_win_per1k"] = d["race_level_win"] / km
    day_rows, loo, loo_win, n_other = [], np.full(len(d), np.nan), np.full(len(d), np.nan), np.zeros(len(d), int)
    for (dt, ven, surf), g in d.groupby(["date", "venue", "surf"]):
        v = g["race_level_per1k"].to_numpy(float)
        day_rows.append({"date": dt, "venue": ven, "surf": surf, "n_races": len(v),
                         "day_level_per1k": round(_shrunk_median(v, k), 4), "day_level_raw_median_per1k": round(float(np.median(v)), 4)})
        vw = g["race_level_win_per1k"].to_numpy(float) if "race_level_win_per1k" in g else None
        for j, idx in enumerate(g.index):
            others = np.delete(v, j)
            n_other[idx] = len(others)
            loo[idx] = _shrunk_median(others, k)
            if vw is not None:
                loo_win[idx] = _shrunk_median(np.delete(vw, j), k)
    d["day_level_loo_per1k"] = loo
    d["n_other_races"] = n_other
    d["member_level_per1k"] = d["race_level_per1k"] - d["day_level_loo_per1k"]
    d["member_level"] = d["member_level_per1k"] * km
    if win_model is not None:
        d["member_level_win_per1k"] = d["race_level_win_per1k"] - loo_win
    days = pd.DataFrame(day_rows)
    days["day_word"] = days["day_level_per1k"].map(lambda x: word(x, TRACK_WORDS))
    d["race_word"] = d["race_level_per1k"].map(lambda x: word(x, TRACK_WORDS))
    d["member_word"] = d["member_level_per1k"].map(lambda x: word(x, MEMBER_WORDS))
    d = d.merge(days[["date", "venue", "surf", "day_level_per1k", "day_word"]], on=["date", "venue", "surf"], how="left")
    d["day_level"] = d["day_level_per1k"] * km
    keep = ["race_id", "date", "venue", "surf", "distance", "course", "class_ord", "age_ord", "newcomer", "field_size", "front600",
            "pace_z", "pace_label", "y_top5", "y_win", "t_hat", "race_level", "race_level_raw", "race_level_per1k", "race_level_raw_per1k",
            "race_word", "day_level", "day_level_per1k", "day_word", "day_level_loo_per1k", "n_other_races", "member_level",
            "member_level_per1k", "member_word"] + (["race_level_win", "member_level_win_per1k"] if win_model is not None else [])
    for c in d.columns:
        if d[c].dtype == float:
            d[c] = d[c].round(4)
    return d[keep], days


def view_of(row: dict[str, Any] | None, day: dict[str, Any] | None, meta: dict[str, Any]) -> dict[str, Any]:
    """API に載せる形（レース結果の欄の 3 行と注）。値が無い所は None（画面は「算出なし」）。"""
    if not row:
        return {"available": False, "note": "算出なし", "version": meta.get("version")}
    sec = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 1)   # noqa: E731
    ok_member = (row.get("n_other_races") or 0) >= 2
    return {
        "available": True, "version": meta.get("version"), "built": meta.get("built"),
        "race": {"sec": sec(row.get("race_level")), "raw_sec": sec(row.get("race_level_raw")), "word": row.get("race_word"),
                 "pace": row.get("pace_label")},
        "day": {"sec": sec(row.get("day_level")), "word": row.get("day_word"), "venue": row.get("venue"), "surf": row.get("surf"),
                "n_races": (day or {}).get("n_races")},
        "member": {"sec": sec(row.get("member_level")) if ok_member else None, "word": row.get("member_word") if ok_member else None},
        "formula": "基準の時計 T̂＝コース × クラス × 年齢の条件 × 頭数 × 前半 600m の単調な関数（2016〜24・v2）。"
                   "レース＝上位 5 頭の時計の中央値 − T̂、開催日＝その日・場・芝ダのレースの中央値（件数で縮める）、"
                   "メンバー＝レース − 自分のレースを除いた開催日。言葉の段は 1000m あたりの秒で決める。",
    }


# ── 読み出し（API。手元の parquet を読むだけ。表示のときに計算しない） ──
import json as _json
import threading as _threading
from pathlib import Path as _Path

LEVEL_REL = "data/page_reference/race_level"
_REPO = _Path(__file__).resolve().parents[3]
_cache: dict[str, tuple[float, Any]] = {}
_lock = _threading.Lock()


REFRESH_SEC = 6 * 3600.0     # 取ってからこれだけ経ったら取り直す（ホストで作り直した値を、立ち上がっている台にも届ける）
MISS_TTL_SEC = 3600.0        # GCS に無かったら、この秒数は問い合わせない


def _local(rel: str) -> _Path | None:
    """手元のファイル。無い・古い（REFRESH_SEC 超）ときは、ミラーの GCS から 1 件だけ取る。
    今（2026-11-26〜）は 3 ファイルを手で GCS に置き、Cloud Run は読まれたときに取る。12/1 以降はホストの毎日の実行が書き、
    gcs_mirror の TREES（束と目録）に足す（このとき束に入るので、ここの取りは起動直後の 1 回だけになる）。
    費用（CLAUDE.md の最悪の回数）: 1 台あたり、使う年のファイル 3 件（meta・races_<年>・days_<年>）× 6 時間に 1 回の取り直し。
    最悪 起動 96 回/日 × 3 件 ＋ 取り直し（1 台 4 回/日 × 3 件）≒ 300 回/日 × 30 日 ≒ 9,000 回/月（Class B 約 0.06 円/千回 → 約 0.5 円/月）。
    GCS に無いときは MISS_TTL_SEC ごとに 1 回（最悪 24 回/日 × 3 件/台）。東京の中の読み出しなので転送料は無い。"""
    import time as _time

    p = _REPO / rel
    if p.is_file() and _time.time() - p.stat().st_mtime < REFRESH_SEC:
        return p
    try:
        from src.ops.gcs_mirror import fetch_one

        fetch_one(rel, miss_ttl=MISS_TTL_SEC, max_age=REFRESH_SEC)
    except Exception:  # noqa: BLE001
        pass
    return p if p.is_file() else None


def _read(rel: str, kind: str) -> Any:
    p = _local(rel)
    if p is None:
        return None
    mt = p.stat().st_mtime
    with _lock:
        hit = _cache.get(rel)
        if hit and hit[0] == mt:
            return hit[1]
    val = _json.loads(p.read_text(encoding="utf-8")) if kind == "json" else pd.read_parquet(p)
    with _lock:
        _cache[rel] = (mt, val)
    return val


def result_block(race_id: str, date: str) -> dict[str, Any]:
    """レース結果の API（/api/race/{id}/result）に足す塊。値が無ければ available=False（画面は「算出なし」）。"""
    y = str(date or "").replace("-", "")[:4]
    meta = _read(f"{LEVEL_REL}/meta.json", "json") or {}
    races = _read(f"{LEVEL_REL}/races_{y}.parquet", "parquet") if y else None
    days = _read(f"{LEVEL_REL}/days_{y}.parquet", "parquet") if y else None
    if races is None or races.empty:
        return view_of(None, None, meta)
    hit = races[races["race_id"] == str(race_id)]
    if hit.empty:
        return view_of(None, None, meta)
    row = hit.iloc[0].to_dict()
    day = None
    if days is not None and not days.empty:
        dh = days[(days["date"] == row["date"]) & (days["venue"] == row["venue"]) & (days["surf"] == row["surf"])]
        day = dh.iloc[0].to_dict() if not dh.empty else None
    out = view_of(row, day, meta)
    if not meta.get("member_visible", False):
        out["member"] = {"sec": None, "word": None}
    return out
