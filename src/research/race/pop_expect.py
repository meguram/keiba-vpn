"""人気を考えた「好走の期待得点」。

走法マップの好走ゾーンは、着順の得点（running_map_course.score_of）を、でたらめな着順の平均（頭数だけで決まる）と比べていた。
本来能力が足りない人気薄が適性で好走した／能力が高い人気馬が適性外で凡走した走を強く拾うため、
期待得点を「頭数と人気順を同じにした馬の平均得点」に置き換える。人気馬の凡走は大きなマイナス、人気薄の好走は大きなプラスになる。
人気は確定の人気順（race_result_flat.popularity）。表は過去の全走から作る小さな表（頭数 5〜18 × 人気順）。"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

TABLE_DIR = Path("data/page_reference/tables")
N_MIN, N_MAX = 5, 18
SHRINK = 40.0    # 走数の少ないセルは隣の人気順の平均へ寄せる（疑似走数）


def _score(pos: np.ndarray) -> np.ndarray:
    from src.research.race.running_map_course import score_of
    return score_of(pos)


def popularity_frame(years: Iterable[int]) -> pd.DataFrame:
    """race_id, horse_id, popularity, n_field, finish_position（確定の人気と着順がそろう走）。"""
    parts = []
    for y in years:
        p = TABLE_DIR / str(y) / "race_result_flat.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["race_id", "horse_id", "popularity", "finish_position"])
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["race_id", "horse_id", "popularity", "n_field", "finish_position"])
    d = pd.concat(parts, ignore_index=True)
    d["race_id"] = d["race_id"].astype(str)
    d["horse_id"] = d["horse_id"].astype(str)
    d["popularity"] = pd.to_numeric(d["popularity"], errors="coerce")
    d["finish_position"] = pd.to_numeric(d["finish_position"], errors="coerce")
    d["n_field"] = d.groupby("race_id")["race_id"].transform("size")
    return d.drop_duplicates(["race_id", "horse_id"])


def _pav_decreasing(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """人気順が下がるほど期待得点が下がる（単調非増加）ように均す（PAV）。"""
    blocks: list[list[float]] = []
    for x, wt in zip(v, w):
        blocks.append([x * wt, wt, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop(); blocks[-1][0] += b[0]; blocks[-1][1] += b[1]; blocks[-1][2] += b[2]
    out = []
    for s, wt, k in blocks:
        out += [s / wt] * int(k)
    return np.array(out)


def build_table(frame: pd.DataFrame) -> dict[str, list[float]]:
    """{ "n": [pop=1 の期待得点, 2, ...] }。frame は popularity_frame の出力（学習に使う期間だけ）。"""
    f = frame.dropna(subset=["popularity", "finish_position"]).copy()
    f = f[(f["n_field"] >= N_MIN) & (f["popularity"] >= 1) & (f["popularity"] <= f["n_field"])]
    f["y"] = _score(f["finish_position"].to_numpy())
    f["nb"] = f["n_field"].clip(N_MIN, N_MAX).astype(int)
    table: dict[str, list[float]] = {}
    for n in range(N_MIN, N_MAX + 1):
        g = f[f["nb"] == n]
        pops = np.arange(1, n + 1)
        s = np.array([g.loc[g["popularity"] == k, "y"].sum() for k in pops])
        c = np.array([(g["popularity"] == k).sum() for k in pops], float)
        mean_all = g["y"].mean() if len(g) else float(_score(np.arange(1, n + 1)).mean())
        raw = np.where(c > 0, s / np.maximum(c, 1), mean_all)
        nb = np.convolve(np.pad(raw, 1, mode="edge"), [1 / 3, 1 / 3, 1 / 3], mode="valid")   # 隣の人気順の平均
        m = (s + SHRINK * nb) / (c + SHRINK)
        table[str(n)] = [round(float(v), 4) for v in _pav_decreasing(m, c + SHRINK)]
    return table


def expect(table: Optional[dict], n_field: np.ndarray, popularity: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """頭数と人気順から期待得点。人気が無い走・表が無いときは fallback（頭数だけの期待得点）。"""
    n = np.asarray(n_field, dtype=float); p = np.asarray(popularity, dtype=float)
    out = np.array(fallback, dtype=float)
    if not table:
        return out
    for i in range(len(out)):
        if not (p[i] >= 1) or not (n[i] >= N_MIN):
            continue
        row = table[str(int(min(max(n[i], N_MIN), N_MAX)))]
        k = int(min(p[i], len(row)))
        out[i] = row[k - 1]
    return out


@lru_cache(maxsize=1)
def default_table() -> Optional[dict]:
    """全期間の表（本番用）。置き場 data/page_reference/pop_expect.json（無ければ作る）。"""
    path = TABLE_DIR.parent / "pop_expect.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    years = sorted(int(p.name) for p in TABLE_DIR.iterdir() if p.is_dir() and p.name.isdigit())
    tab = build_table(popularity_frame(years))
    try:
        path.write_text(json.dumps(tab), encoding="utf-8")
    except OSError:
        pass
    return tab
