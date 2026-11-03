"""
走法マップの「その日の馬場傾向」（同じ日・同じ競馬場・同じ芝ダートの、このレースより前のレースの結果から）。

ユーザー指示（2026-11-02）: 統計的なカラーマップ（コースの全期間）とは別に、その日の馬場の傾向（芝/ダート別）を反映したカラーマップを出す。
傾向があるときだけ。地図の 8 方面に加えて、地図に入れない軸（芝道悪適性・ダートタフ適正・急坂適性）も対象。

  - 対象の走: race_id の先頭 10 桁（年・場・回・日）が同じで、レース番号が小さく、芝ダートが同じレースの、着順のある馬。
  - 各馬の軸の値は、その日より前の走だけから作ったプロファイル（地図と同じ。その日の結果は入らない）。
  - 好走の得点はコースの地図と同じ（1〜3 着=1、4〜7 着は 0.2 ずつ下げ、8 着以下=0）。期待は頭数から決まる平均。
  - 軸ごとの傾向: 得点 − 期待 を軸の値（z。補助の軸は (値−50)/10）へ単回帰した傾き。標準誤差はレース単位でまとめた
    サンドウィッチ分散（同じレースの馬は独立でないため。レース数 G が少ないので CR1 で補正し、自由度 G−1 の t 分布で p を出す）。軸どうしの多重検定は BH 法（FDR 5%）で補正し、通ったものを「傾向あり」とする。
    対象のレースが MIN_RACES 未満・走が MIN_RUNS 未満なら判定しない。
  - カラーマップ: その日の走をこのレースの地図（コースの 3 方面の座標）に置き、格子ごとに 観測 / 期待 の倍率を出す。
    走が少ないので 1（その日の平均）へ強めに寄せる（M=SHRINK）。有意性は格子ごとのレース単位の z と BH 法（FDR 5%）。
注意: 1 日のレース数は多くても 11 で、偶然の偏りと区別しにくい。「傾向あり」は FDR 5% の目安で（信号の無い日でも約 6% の日に出る）、強い証拠ではない。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.research.race import running_map_aptitude as apt
from src.research.race import running_map_course as rmc

MIN_RACES = 3
MIN_RUNS = 30
FDR = 0.05
SHRINK = 8.0
BANDWIDTH = 0.6


def earlier_race_ids(race_id: str) -> list[str]:
    rid = str(race_id)
    return [f"{rid[:10]}{k:02d}" for k in range(1, int(rid[10:12]))]


def _slope_t(v: np.ndarray, r: np.ndarray, races: np.ndarray) -> tuple[float, float, float, int]:
    """傾き・t・p（自由度 G−1 の t 分布）・走数。G（レース数）が少ないとサンドウィッチ分散は小さく出すぎるので CR1 で補正する。"""
    from scipy.stats import t as tdist

    ok = ~np.isnan(v)
    v, r, races = v[ok], r[ok], races[ok]
    G, N = len(set(races)), len(v)
    if N < MIN_RUNS or G < MIN_RACES:
        return np.nan, np.nan, np.nan, int(N)
    vc = v - v.mean()
    sxx = float((vc ** 2).sum())
    if sxx < 1e-9:
        return np.nan, np.nan, np.nan, int(N)
    b = float((vc * r).sum() / sxx)
    e = r - r.mean() - b * vc
    codes = pd.factorize(races)[0]
    s = np.bincount(codes, weights=vc * e)
    se = float(np.sqrt((s ** 2).sum()) / sxx) * np.sqrt(G / (G - 1) * (N - 1) / (N - 2))
    if se <= 0:
        return b, np.nan, np.nan, int(N)
    tt = b / se
    return b, tt, float(2 * tdist.sf(abs(tt), G - 1)), int(N)


def _bh(p: np.ndarray, q: float) -> np.ndarray:
    sel = np.zeros(len(p), bool)
    ok = ~np.isnan(p)
    if ok.any():
        idx = np.where(ok)[0]
        sel[idx[apt._bh(p[ok], q)]] = True
    return sel


def compute(
    race_id: str,
    conditions: dict[str, Any],
    load_result: Callable[[str], Optional[dict]],
    horse_axes: Callable[[list[str]], pd.DataFrame],
    axis_meta: list[dict[str, Any]],
    course_map: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """その日の馬場傾向。前のレースが無ければ None。

    horse_axes(horse_ids) は、各馬の軸の値（z。列名は axis_meta の key、補助の軸は偏差値のまま）を horse_id 索引で返す。
    axis_meta は [{key, label, high, low, kind: "map"|"extra"}]。"""

    surface = conditions.get("surface")
    rows = []
    for rid in earlier_race_ids(race_id):
        card = load_result(rid)
        if not card or not card.get("entries"):
            continue
        if rmc_surface(card.get("surface")) != surface:
            continue
        ents = [e for e in card["entries"] if e.get("horse_id") and isinstance(e.get("finish_position"), (int, float)) and e["finish_position"] >= 1]
        if len(ents) < rmc.MIN_FIELD:
            continue
        n = len(ents)
        for e in ents:
            rows.append({"race_id": rid, "horse_id": str(e["horse_id"]), "finish_position": int(e["finish_position"]), "n_field": n,
                         "track_condition": card.get("track_condition")})
    if not rows:
        return None
    t = pd.DataFrame(rows)
    t["y"] = rmc.score_of(t["finish_position"].to_numpy())
    t["e"] = rmc.expected_score(t["n_field"].to_numpy())
    vals = horse_axes(sorted(t["horse_id"].unique()))
    t = t.join(vals, on="horse_id")
    races = t["race_id"].to_numpy()
    r = (t["y"] - t["e"]).to_numpy()
    n_races = int(t["race_id"].nunique())
    out: dict[str, Any] = {
        "surface": surface, "n_races": n_races, "n_runs": int(len(t)),
        "races": sorted(t["race_id"].unique().tolist()),
        "track_conditions": sorted({c for c in t["track_condition"].dropna().unique()}),
        "axes": [], "trend": [], "heat": None,
    }
    if n_races < MIN_RACES or len(t) < MIN_RUNS:
        out["status"] = "few"
        return out

    stats = []
    for a in axis_meta:
        if a["key"] not in t.columns:
            continue
        v = t[a["key"]].to_numpy(dtype=float)
        if a.get("kind") == "extra":
            v = (v - 50) / 10
        b, tt, pp, nn = _slope_t(v, r, races)
        stats.append({**a, "slope": None if np.isnan(b) else round(b, 4), "t": None if np.isnan(tt) else round(tt, 2),
                      "p": None if np.isnan(pp) else round(pp, 4), "n": nn, "_p": pp})
    sel = _bh(np.array([s["_p"] for s in stats]), FDR)
    for s, ok in zip(stats, sel):
        s["trend"] = bool(ok)
        s.pop("_p")
    out["axes"] = stats
    out["trend"] = [{"key": s["key"], "label": s["label"], "text": s["high"] if s["slope"] > 0 else s["low"], "t": s["t"]}
                    for s in sorted([s for s in stats if s["trend"]], key=lambda s: -abs(s["t"]))]

    if course_map:
        keys = course_map.get("keys") or []
        Zm = t[keys].to_numpy(dtype=float) if keys and all(k in t.columns for k in keys) else np.empty((len(t), 0))
        top = [keys.index(k) for k in course_map["axes"]] if Zm.shape[1] else []
        ok = ~np.isnan(Zm[:, top]).any(axis=1) if top else np.zeros(len(t), bool)      # 地図の上位の方面がすべてある走だけ
        if ok.sum() >= MIN_RUNS:
            P = rmc.place(Zm[ok], course_map)
            out["heat"] = _heat(P, t["y"].to_numpy()[ok], t["e"].to_numpy()[ok], pd.factorize(races[ok])[0], course_map["extent"])
    out["status"] = "trend" if out["trend"] or (out["heat"] and "2" in out["heat"]["sig"]) else "none"
    return out


def _heat(P: np.ndarray, y: np.ndarray, e: np.ndarray, codes: np.ndarray, extent: dict[str, Any]) -> dict[str, Any]:
    from scipy.stats import t as tdist

    grid = apt._grid(extent)
    d2 = ((grid[:, None, :] - P[None, :, :]) ** 2).sum(2)
    W = np.exp(-d2 / (2 * BANDWIDTH ** 2))
    base = float(y.sum() / max(e.sum(), 1e-9))
    o, x = W @ y, W @ (e * base)
    lift = (o + SHRINK) / (x + SHRINK)
    valid = x > 1.5
    res = y - e * base
    nr = int(codes.max()) + 1
    tr = np.stack([np.bincount(codes, weights=W[j] * res, minlength=nr) for j in range(len(grid))])
    z = tr.sum(1) / np.sqrt((tr ** 2).sum(1) + 1e-9)
    sel = np.zeros(len(grid), bool)
    if valid.any():
        sel[np.where(valid)[0][apt._bh(2 * tdist.sf(np.abs(z[valid]), max(nr - 1, 1)), FDR)]] = True   # レース数が少ないので t(G−1)
    sig = np.where(~valid, 0, np.where(sel & (z > 0), 2, 1))
    return {"lift": [round(float(v), 3) for v in lift], "sig": "".join(str(int(v)) for v in sig)}


def rmc_surface(s: Any) -> Optional[str]:
    s = str(s or "")
    return "芝" if s.startswith("芝") else ("ダート" if s.startswith("ダ") else None)
