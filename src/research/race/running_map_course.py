"""
走法マップのコース別の地図（コースで重要な 3 軸だけで作る地図と、その座標での好走しやすさ）。

ユーザー指示（2026-11-02）: コースごとに重要度の高い軸を 3 つに絞って地図にし、どのゾーンが好走しやすいかを事前に計算する。

好走の得点（ユーザー指定）: 1〜3 着 = 1、4 着 0.8、5 着 0.6、6 着 0.4、7 着 0.2、8 着以下 = 0。
期待は「頭数 n のレースで着順がでたらめに決まったときの得点の平均」= Σ_{k=1..n} score(k) / n。

手順（基準を作るときに 1 回だけ。running_map._build_reference から呼ぶ）:
  1. 対象: 2024-01-01 以降の平地（頭数 5 以上）。各走の 8 方面は「その走より前の走」から作ったプロファイル（profile_table）を、
     全馬の分布で順位 → 標準正規にそろえた値（地図と同じ z）。8 方面のうち値があるのが 5 未満の走は使わない。欠けは 0（全馬の中央値）。
  2. 重要度: コースの走で、得点 − 期待 を 8 方面へ線形回帰し、方面ごとに「その方面を抜いたときの R² の落ち」を重要度とする
     （他の方面と重なる分は数えないので、先行力と Hペース耐性のように似た方面が両方選ばれにくい）。上位 3 つを地図の軸にする。
  3. 地図: その 3 方面（3 つとも実測由来の値がある走）の主成分で 2 次元に置く。向きは、最も重要な方面の矢印が真上、
     先行力が入っていれば先行力が左（無ければ 2 番目の方面が右）。
  4. 好走しやすさ: 地図の 24×24 の格子で、ガウスカーネル（幅 0.5）で近くの走を平滑化し、観測した得点 / 期待 の倍率を出す。
     サンプルの少ないゾーンはコース全体の倍率へ寄せる（M=20）。有意性はレース単位でまとめたサンドウィッチ分散の z と BH 法（FDR 5%）。
  コースの走が MIN_RUNS 未満なら、同じ芝ダ・距離帯（≤1400 / 〜1700 / 1800+）にまとめた「グループ」で作り、そのコースはグループの地図を使う。

注意: 重要度・倍率は 2024 年以降の全レースで測った値（標本内）。2024 年の走は位置の 4 方面（Hペース耐性・先行力・Cr性能・まくり特性）の
実測が少なく（約 13%）、その分は前の走が無く欠け（0 = 中央値）になりやすい。
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.research.race import running_map_aptitude as apt

SINCE = "2024-01-01"
TOP_K = 3
MIN_AXES_PRESENT = 5
MIN_RUNS = 600
MIN_FIELD = 5


def score_of(pos: np.ndarray) -> np.ndarray:
    """着順 → 好走の得点（1〜3 着 = 1、4〜7 着は 0.2 ずつ下げ、8 着以下 = 0）。"""
    pos = np.asarray(pos, dtype=float)
    return np.where(pos >= 1, np.clip((8 - pos) / 5, 0, 1), 0.0)


def expected_score(n: np.ndarray) -> np.ndarray:
    """頭数 n で着順がでたらめなときの得点の平均。"""
    table = {k: float(score_of(np.arange(1, k + 1)).mean()) for k in range(1, 41)}
    return np.array([table.get(int(v), table[40]) for v in n])


def band_of(distance: Any) -> str:
    d = int(distance)
    return "短" if d <= 1400 else ("マイル" if d < 1800 else "中長")


def group_key(surface: Any, distance: Any) -> str:
    return f"{surface}|{band_of(distance)}"


def _importance(Z: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """r を Z へ線形回帰。方面ごとの R² の落ち・係数・全体の R²。"""
    X = np.column_stack([np.ones(len(Z)), Z])
    tss = float(((r - r.mean()) ** 2).sum()) or 1e-12

    def r2(cols: list[int]) -> tuple[float, np.ndarray]:
        Xs = X[:, [0] + [c + 1 for c in cols]]
        beta, *_ = np.linalg.lstsq(Xs, r, rcond=None)
        return 1 - float(((r - Xs @ beta) ** 2).sum()) / tss, beta

    k = Z.shape[1]
    full, beta = r2(list(range(k)))
    drop = np.array([full - r2([c for c in range(k) if c != j])[0] for j in range(k)])
    return np.maximum(drop, 0), beta[1:], full


def _orient(Y: np.ndarray, V: np.ndarray, axis_keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """最も重要な方面（V の 0 行目）を真上に。先行力があれば左、無ければ 2 番目の方面を右へ。"""
    up = V[0]
    if np.linalg.norm(up) > 1e-9:
        rot = math.pi / 2 - math.atan2(up[1], up[0])
        R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
        Y, V = Y @ R.T, V @ R.T
    if "front" in axis_keys[1:]:
        if V[axis_keys.index("front")][0] > 0:
            Y, V = Y * [-1, 1], V * [-1, 1]
    elif V[1][0] < 0:
        Y, V = Y * [-1, 1], V * [-1, 1]
    return Y, V


def _heat(P: np.ndarray, y: np.ndarray, e: np.ndarray, race_codes: np.ndarray, extent: dict[str, Any]) -> dict[str, Any]:
    from scipy.stats import norm

    grid = apt._grid(extent)
    co, cx = apt._obs_exp(grid, P, y, e)
    base = float(y.sum() / max(e.sum(), 1e-9))
    valid = cx > apt.MIN_EXPECTED
    z = apt._cluster_z(grid, P, y - e * base, race_codes)
    sel = np.zeros(len(grid), bool)
    if valid.any():
        sel[np.where(valid)[0][apt._bh(2 * norm.sf(np.abs(z[valid])))]] = True
    lift = (co + apt.SHRINK * base) / (cx + apt.SHRINK) / base   # コース全体を 1 とした倍率
    sig = np.where(~valid, 0, np.where(sel & (z > 0), 2, 1))
    return {"lift": [round(float(v), 3) for v in lift], "sig": "".join(str(int(v)) for v in sig), "base": round(base, 3)}


def _one(g: pd.DataFrame, Zall: np.ndarray, present: np.ndarray, axis_keys: list[str]) -> Optional[dict[str, Any]]:
    idx = g.index.to_numpy()
    Z, pr = Zall[idx], present[idx]
    r = g["y"].to_numpy() - g["e"].to_numpy()
    drop, beta, full = _importance(Z, r)
    order = list(np.argsort(-drop)[:TOP_K])
    keys = [axis_keys[j] for j in order]
    ok = pr[:, order].all(axis=1)
    if ok.sum() < 200:
        return None
    Zs = Z[ok][:, order]
    mu = Zs.mean(0)
    Zc = Zs - mu
    _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    Y, A = _orient(Zc @ Vt[:2].T, Vt[:2].T, keys)
    recon = Y @ A.T
    var = Zc.var(0)
    extent = apt.extent_from(Y)
    sub = g[ok]
    heat = _heat(Y, sub["y"].to_numpy(), sub["e"].to_numpy(), pd.factorize(sub["race_id"])[0], extent)
    share = drop / drop.sum() if drop.sum() > 0 else drop
    return {
        "axes": keys,
        "importance": [{"key": axis_keys[j], "share": round(float(share[j]), 3), "r2_drop": round(float(drop[j]), 5),
                        "sign": int(np.sign(beta[j])) if beta[j] != 0 else 0} for j in np.argsort(-drop)],
        "r2_full": round(float(full), 4),
        "mu": mu.tolist(), "A": A.tolist(),
        "r2": [None if v < 1e-12 else round(float(1 - (Zc[:, j] - recon[:, j]).var() / v), 3) for j, v in enumerate(var)],
        "explained": round(float((S[:2] ** 2).sum() / (S ** 2).sum()), 3),
        "extent": extent,
        "n_runs": int(len(sub)), "n_races": int(sub["race_id"].nunique()),
        **heat,
    }


def build(tab: pd.DataFrame, Z: np.ndarray, present: np.ndarray, axis_keys: list[str]) -> dict[str, Any]:
    """tab: race_id, date, venue, surface, distance, finish_position, n_field（Z・present と同じ並び）。
    Z: 8 方面の z（地図と同じ変換・符号。欠けは NaN）。present: 実測由来の値があるか。"""
    t = tab.reset_index(drop=True).copy()
    t["y"] = score_of(t["finish_position"].to_numpy())
    t["e"] = expected_score(t["n_field"].to_numpy())
    keep = ((t["date"] >= pd.Timestamp(SINCE)) & (t["n_field"] >= MIN_FIELD) & t[["venue", "surface", "distance"]].notna().all(axis=1)
            & (present.sum(axis=1) >= MIN_AXES_PRESENT) & t["finish_position"].notna()).to_numpy()
    t = t[keep].reset_index(drop=True)
    Zk = np.nan_to_num(Z[keep])
    pk = present[keep]

    courses: dict[str, Any] = {}
    groups: dict[str, Any] = {}
    t["ckey"] = [apt.course_key(v, s, d) for v, s, d in zip(t["venue"], t["surface"], t["distance"])]
    t["gkey"] = [group_key(s, d) for s, d in zip(t["surface"], t["distance"])]
    for gk, g in t.groupby("gkey"):
        if len(g) >= MIN_RUNS:
            res = _one(g, Zk, pk, axis_keys)
            if res:
                groups[gk] = {**res, "level": "group", "label": gk.replace("|", " ")}
    for ck, g in t.groupby("ckey"):
        if len(g) >= MIN_RUNS:
            res = _one(g, Zk, pk, axis_keys)
            if res:
                courses[ck] = {**res, "level": "course"}
    return {"since": SINCE, "score": "1〜3着=1, 4着0.8, 5着0.6, 6着0.4, 7着0.2, 8着以下=0", "courses": courses, "groups": groups}


def lookup(maps: dict[str, Any], venue: Any, surface: Any, distance: Any) -> Optional[dict[str, Any]]:
    """コースの地図。コースが無ければ同じ芝ダ・距離帯のグループの地図。どちらも無ければ None。"""
    if not (maps and venue and surface and distance):
        return None
    hit = maps.get("courses", {}).get(apt.course_key(venue, surface, distance))
    if hit:
        return hit
    return maps.get("groups", {}).get(group_key(surface, distance))
